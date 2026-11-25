
from pathlib import Path
import random
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from model import IndoBERTClassifier, ID2LABEL

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKENIZED_DIR = PROJECT_ROOT / "data" / "processed" / "tokenized_audited_v3"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics_audited_v3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
MAX_EPOCHS = 5
PATIENCE = 2
LEARNING_RATES = [2e-5, 3e-5]
BATCH_SIZES = [16, 32]
NUM_LABELS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_tokenized_data():
    train_data = torch.load(TOKENIZED_DIR / "train.pt", map_location="cpu", weights_only=True)
    val_data = torch.load(TOKENIZED_DIR / "validation.pt", map_location="cpu", weights_only=True)

    input_ids = torch.cat([train_data["input_ids"], val_data["input_ids"]], dim=0)
    attention_mask = torch.cat([train_data["attention_mask"], val_data["attention_mask"]], dim=0)
    labels = torch.cat([train_data["labels"], val_data["labels"]], dim=0)
    comment_ids = train_data["comment_ids"] + val_data["comment_ids"]

    return input_ids, attention_mask, labels, comment_ids

def create_loader(input_ids, attention_mask, labels, batch_size, shuffle):
    dataset = TensorDataset(input_ids, attention_mask, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )

def get_optimizer_grouped_parameters(model, encoder_lr, head_lr, weight_decay=0.03):
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = []

    for name, param in model.indobert.named_parameters():
        if not param.requires_grad:
            continue
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        lr = encoder_lr * 0.6 if any(f"encoder.layer.{i}." in name for i in range(6)) else encoder_lr
        optimizer_grouped_parameters.append({"params": [param], "lr": lr, "weight_decay": wd})

    for module in [model.layer_pooler, model.layer_norm, model.classifier]:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
            optimizer_grouped_parameters.append({"params": [param], "lr": head_lr, "weight_decay": wd})

    return optimizer_grouped_parameters

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits.float(), labels)

            total_loss += loss.item()
            probs = torch.softmax(logits.float(), dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.argmax(all_probs, axis=1)

    return {
        "val_loss": total_loss / len(loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision_macro": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall_macro": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "probs": all_probs,
        "preds": all_preds,
        "labels": np.array(all_labels),
    }

def train_one_fold(input_ids, attention_mask, labels, train_idx, val_idx, lr, batch_size, fold_num):
    set_seed(RANDOM_STATE + fold_num)

    train_loader = create_loader(input_ids[train_idx], attention_mask[train_idx], labels[train_idx], batch_size, True)
    val_loader = create_loader(input_ids[val_idx], attention_mask[val_idx], labels[val_idx], batch_size, False)

    model = IndoBERTClassifier(num_labels=NUM_LABELS, dropout_rate=0.2).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = AdamW(get_optimizer_grouped_parameters(model, encoder_lr=lr, head_lr=1e-4, weight_decay=0.03))

    total_steps = len(train_loader) * MAX_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_f1 = -1.0
    best_metrics = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_train_loss = 0.0

        for input_ids_b, attention_mask_b, labels_b in train_loader:
            input_ids_b = input_ids_b.to(DEVICE)
            attention_mask_b = attention_mask_b.to(DEVICE)
            labels_b = labels_b.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                logits = model(input_ids=input_ids_b, attention_mask=attention_mask_b)
                loss = criterion(logits.float(), labels_b)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, criterion)

        print(
            f"Fold {fold_num} | Epoch {epoch}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val F1: {val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_metrics = val_metrics
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping dijalankan.")
            break

    del model, optimizer, scaler
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return best_epoch, best_metrics

def main():
    set_seed()
    input_ids, attention_mask, labels, comment_ids = load_tokenized_data()
    label_array = labels.numpy()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    experiments = list(itertools.product(LEARNING_RATES, BATCH_SIZES))
    best_overall_f1 = -1.0
    best_oof_df = None
    experiment_results = []

    for exp_num, (lr, batch_size) in enumerate(experiments, start=1):
        print("\n" + "=" * 60)
        print(f"EXPERIMENT {exp_num}/{len(experiments)} | Encoder LR: {lr} | Head LR: 1e-4 | Batch Size: {batch_size}")
        print("=" * 60)

        oof_probs = np.zeros((len(label_array), NUM_LABELS))
        fold_summaries = []

        for fold_num, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(label_array)), label_array), start=1):
            print(f"\nMemulai Fold {fold_num}/{N_SPLITS}")
            best_epoch, best_metrics = train_one_fold(
                input_ids, attention_mask, labels, train_idx, val_idx, lr, batch_size, fold_num
            )
            oof_probs[val_idx] = best_metrics["probs"]
            fold_summaries.append({
                "best_epoch": best_epoch,
                "accuracy": best_metrics["accuracy"],
                "precision_macro": best_metrics["precision_macro"],
                "recall_macro": best_metrics["recall_macro"],
                "f1_macro": best_metrics["f1_macro"],
            })

        oof_preds = np.argmax(oof_probs, axis=1)
        oof_f1 = f1_score(label_array, oof_preds, average="macro")

        print(f"\n[OOF Classification Report - Exp {exp_num}]")
        print(classification_report(label_array, oof_preds, target_names=["Negatif", "Netral", "Positif"], digits=4))

        df_sum = pd.DataFrame(fold_summaries)
        summary = {
            "experiment": exp_num,
            "learning_rate": lr,
            "batch_size": batch_size,
            "mean_f1_macro": df_sum["f1_macro"].mean(),
            "oof_f1_macro": oof_f1,
            "mean_best_epoch": df_sum["best_epoch"].mean(),
        }
        experiment_results.append(summary)

        if oof_f1 > best_overall_f1:
            best_overall_f1 = oof_f1
            best_oof_df = pd.DataFrame({
                "comment_id": comment_ids,
                "true_label": [ID2LABEL[i] for i in label_array],
                "pred_label": [ID2LABEL[i] for i in oof_preds],
                "prob_Negatif": oof_probs[:, 0],
                "prob_Netral": oof_probs[:, 1],
                "prob_Positif": oof_probs[:, 2],
            })
            best_oof_df["true_label_prob"] = [
                oof_probs[idx, true_cls] for idx, true_cls in enumerate(label_array)
            ]

    pd.DataFrame(experiment_results).to_csv(RESULTS_DIR / "hyperparameter_results.csv", index=False)
    best_oof_df.to_csv(RESULTS_DIR / "oof_predictions.csv", index=False)
    print("\nSelesai! File OOF predictions disimpan di:", RESULTS_DIR / "oof_predictions.csv")

if __name__ == "__main__":
    main()
