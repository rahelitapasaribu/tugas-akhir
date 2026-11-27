from pathlib import Path
import random
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
    confusion_matrix,
)
from model import IndoBERTClassifier, ID2LABEL

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKENIZED_DIR = PROJECT_ROOT / "data" / "processed" / "tokenized_audited_v3_clean"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics_v3_fgm_ensemble"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 1024]
N_SPLITS = 5
MAX_EPOCHS = 5
ENCODER_LR = 2e-5
HEAD_LR = 1e-4
BATCH_SIZE = 32
NUM_LABELS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


class FGM:
    """Fast Gradient Method untuk regularisasi adversarial pada word embeddings IndoBERT."""
    def __init__(self, model, emb_name="word_embeddings", epsilon=0.5):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_optimizer_grouped_parameters(model, encoder_lr, head_lr, weight_decay=0.04):
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    grouped = []
    for name, param in model.indobert.named_parameters():
        if not param.requires_grad:
            continue
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        lr = encoder_lr * 0.6 if any(f"encoder.layer.{i}." in name for i in range(6)) else encoder_lr
        grouped.append({"params": [param], "lr": lr, "weight_decay": wd})

    for module in [model.layer_pooler, model.layer_norm, model.classifier]:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
            grouped.append({"params": [param], "lr": head_lr, "weight_decay": wd})
    return grouped


@torch.no_grad()
def predict_probs(model, loader, criterion=None):
    model.eval()
    total_loss = 0.0
    all_probs = []
    for batch in loader:
        input_ids = batch[0].to(DEVICE)
        attention_mask = batch[1].to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            if criterion is not None:
                labels = batch[2].to(DEVICE)
                total_loss += criterion(logits.float(), labels).item()
        all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    avg_loss = total_loss / len(loader) if criterion is not None else 0.0
    return np.vstack(all_probs), avg_loss


def main():
    train_data = torch.load(TOKENIZED_DIR / "train.pt", map_location="cpu", weights_only=True)
    val_data = torch.load(TOKENIZED_DIR / "validation.pt", map_location="cpu", weights_only=True)
    test_data = torch.load(TOKENIZED_DIR / "test.pt", map_location="cpu", weights_only=True)

    input_ids = torch.cat([train_data["input_ids"], val_data["input_ids"]], dim=0)
    attention_mask = torch.cat([train_data["attention_mask"], val_data["attention_mask"]], dim=0)
    labels = torch.cat([train_data["labels"], val_data["labels"]], dim=0)
    label_array = labels.numpy()

    test_loader = DataLoader(
        TensorDataset(test_data["input_ids"], test_data["attention_mask"]),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    y_test = test_data["labels"].numpy()

    total_oof_probs = np.zeros((len(label_array), NUM_LABELS))
    total_test_probs = np.zeros((len(y_test), NUM_LABELS))

    for seed in SEEDS:
        print("\n" + "=" * 60)
        print(f"TRAINING 5-FOLD INDOBERT + FGM | SEED {seed}")
        print("=" * 60)
        set_seed(seed)
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        seed_oof = np.zeros((len(label_array), NUM_LABELS))
        seed_test = np.zeros((len(y_test), NUM_LABELS))

        for fold_num, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(label_array)), label_array), start=1):
            set_seed(seed + fold_num)
            train_loader = DataLoader(
                TensorDataset(input_ids[train_idx], attention_mask[train_idx], labels[train_idx]),
                batch_size=BATCH_SIZE,
                shuffle=True,
                pin_memory=USE_AMP,
            )
            val_loader = DataLoader(
                TensorDataset(input_ids[val_idx], attention_mask[val_idx], labels[val_idx]),
                batch_size=BATCH_SIZE,
                shuffle=False,
                pin_memory=USE_AMP,
            )

            model = IndoBERTClassifier(num_labels=NUM_LABELS, dropout_rate=0.25).to(DEVICE)
            fgm = FGM(model, epsilon=0.5)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
            optimizer = AdamW(get_optimizer_grouped_parameters(model, ENCODER_LR, HEAD_LR, weight_decay=0.04))

            total_steps = len(train_loader) * MAX_EPOCHS
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
            )
            scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

            best_score = -1.0
            best_val_probs = None
            best_state = None

            for epoch in range(1, MAX_EPOCHS + 1):
                model.train()
                for ids_b, mask_b, lbl_b in train_loader:
                    ids_b, mask_b, lbl_b = ids_b.to(DEVICE), mask_b.to(DEVICE), lbl_b.to(DEVICE)
                    optimizer.zero_grad(set_to_none=True)

                    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                        logits = model(input_ids=ids_b, attention_mask=mask_b)
                        loss = criterion(logits.float(), lbl_b)

                    scaler.scale(loss).backward()

                    # Adversarial FGM pass
                    fgm.attack()
                    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                        logits_adv = model(input_ids=ids_b, attention_mask=mask_b)
                        loss_adv = criterion(logits_adv.float(), lbl_b)
                    scaler.scale(loss_adv).backward()
                    fgm.restore()

                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= old_scale:
                        scheduler.step()

                val_probs, val_loss = predict_probs(model, val_loader, criterion)
                val_f1 = f1_score(label_array[val_idx], np.argmax(val_probs, axis=1), average="macro")
                # Composite score mencegah pemilihan epoch yang overfit
                comp_score = val_f1 - 0.10 * val_loss
                print(f"Seed {seed} | Fold {fold_num} | Ep {epoch} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

                if comp_score > best_score:
                    best_score = comp_score
                    best_val_probs = val_probs
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}

            seed_oof[val_idx] = best_val_probs
            model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
            fold_test_probs, _ = predict_probs(model, test_loader)
            seed_test += fold_test_probs / N_SPLITS

            del model, optimizer, scaler
            if USE_AMP:
                torch.cuda.empty_cache()

        total_oof_probs += seed_oof / len(SEEDS)
        total_test_probs += seed_test / len(SEEDS)

    # Cari threshold stabil dari gabungan OOF 2-Seed
    best_oof_score, best_w = -1.0, np.array([1.0, 1.0, 1.0])
    for w1 in np.linspace(0.90, 1.20, 31):
        for w2 in np.linspace(0.90, 1.20, 31):
            w = np.array([1.0, w1, w2])
            p = np.argmax(total_oof_probs * w, axis=1)
            sc = 0.5 * accuracy_score(label_array, p) + 0.5 * f1_score(label_array, p, average="macro")
            if sc > best_oof_score:
                best_oof_score, best_w = sc, w

    oof_preds = np.argmax(total_oof_probs * best_w, axis=1)
    print("\n" + "=" * 60)
    print(f"2-SEED OOF REPORT (Optimal Weights: {np.round(best_w, 3)})")
    print("=" * 60)
    print(classification_report(label_array, oof_preds, target_names=["Negatif", "Netral", "Positif"], digits=4))

    test_preds_raw = np.argmax(total_test_probs, axis=1)
    test_preds_tuned = np.argmax(total_test_probs * best_w, axis=1)

    print("=" * 60)
    print("GOLD-STANDARD TEST SET | 10-Model IndoBERT FGM Ensemble (Raw)")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, test_preds_raw):.4f} | Macro F1: {f1_score(y_test, test_preds_raw, average='macro'):.4f}")
    print(classification_report(y_test, test_preds_raw, target_names=["Negatif", "Netral", "Positif"], digits=4))

    print("=" * 60)
    print(f"GOLD-STANDARD TEST SET | 10-Model IndoBERT FGM Ensemble (+ Threshold {np.round(best_w, 3)})")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, test_preds_tuned):.4f} | Macro F1: {f1_score(y_test, test_preds_tuned, average='macro'):.4f}")
    print(classification_report(y_test, test_preds_tuned, target_names=["Negatif", "Netral", "Positif"], digits=4))

    pd.DataFrame({
        "comment_id": test_data["comment_ids"],
        "true_label": [ID2LABEL[i] for i in y_test],
        "pred_label_raw": [ID2LABEL[i] for i in test_preds_raw],
        "pred_label_tuned": [ID2LABEL[i] for i in test_preds_tuned],
        "prob_Negatif": total_test_probs[:, 0],
        "prob_Netral": total_test_probs[:, 1],
        "prob_Positif": total_test_probs[:, 2],
    }).to_csv(RESULTS_DIR / "test_fgm_ensemble_predictions.csv", index=False)


if __name__ == "__main__":
    main()
