"""
Pelatihan 5-Fold Fine-Tuned IndoBERT dengan R-Drop Regularization,
Smooth Layer-Wise Learning Rate Decay, dan Optimasi Macro F1.
"""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from scipy.optimize import minimize_scalar
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
TOKENIZED_DIR = PROJECT_ROOT / "data" / "processed" / "tokenized_v3_parent_ctx"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics_v3_parent_rdrop"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "indobert_parent_rdrop_best"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 2026]
N_SPLITS = 5
MAX_EPOCHS = 5
ENCODER_LR = 3e-5
HEAD_LR = 1.2e-4
BATCH_SIZE = 32
RDROP_ALPHA = 0.5
NUM_LABELS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_rdrop_loss(logits1, logits2, labels, criterion, alpha=RDROP_ALPHA):
    ce_loss = 0.5 * (criterion(logits1, labels) + criterion(logits2, labels))
    p = F.log_softmax(logits1, dim=-1)
    q = F.log_softmax(logits2, dim=-1)
    p_tec = F.softmax(logits1, dim=-1)
    q_tec = F.softmax(logits2, dim=-1)
    kl_loss = 0.5 * (
        F.kl_div(p, q_tec, reduction="batchmean") +
        F.kl_div(q, p_tec, reduction="batchmean")
    )
    return ce_loss + alpha * kl_loss


def get_smooth_llrd_params(model, encoder_lr, head_lr, layer_decay=0.92, weight_decay=0.03):
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    grouped = []
    num_layers = 12

    for name, param in model.indobert.named_parameters():
        if not param.requires_grad:
            continue
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        lr = encoder_lr * (layer_decay ** num_layers)
        for layer_idx in range(num_layers):
            if f"encoder.layer.{layer_idx}." in name:
                lr = encoder_lr * (layer_decay ** (num_layers - 1 - layer_idx))
                break
        grouped.append({"params": [param], "lr": lr, "weight_decay": wd})

    for module in [model.layer_pooler, model.attn_pooler, model.layer_norm, model.classifier]:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
            grouped.append({"params": [param], "lr": head_lr, "weight_decay": wd})

    return grouped


@torch.no_grad()
def predict_logits(model, loader):
    model.eval()
    all_logits = []
    for batch in loader:
        ids = batch[0].to(DEVICE)
        mask = batch[1].to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            logits = model(input_ids=ids, attention_mask=mask)
        all_logits.append(logits.float().cpu().numpy())
    return np.vstack(all_logits)


def optimize_postprocessing(oof_logits, y_true):
    def nll_loss(temp):
        scaled = oof_logits / temp
        log_probs = scaled - np.log(np.sum(np.exp(scaled), axis=1, keepdims=True))
        return -np.mean(log_probs[np.arange(len(y_true)), y_true])

    res_t = minimize_scalar(nll_loss, bounds=(0.5, 2.5), method="bounded")
    best_temp = float(res_t.x)

    scaled_logits = oof_logits / best_temp
    exp_l = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

    weights = np.array([1.0, 1.0, 1.0])
    best_obj = 0.5 * f1_score(y_true, np.argmax(probs, axis=1), average="macro") + 0.5 * accuracy_score(y_true, np.argmax(probs, axis=1))

    for _ in range(4):
        for cls_idx in [1, 2, 0]:
            best_c = weights[cls_idx]
            for cand in np.linspace(0.80, 1.30, 101):
                w_try = weights.copy()
                w_try[cls_idx] = cand
                preds = np.argmax(probs * w_try, axis=1)
                obj = 0.5 * f1_score(y_true, preds, average="macro") + 0.5 * accuracy_score(y_true, preds)
                if obj > best_obj:
                    best_obj = obj
                    best_c = cand
            weights[cls_idx] = best_c

    weights = weights / weights[0]
    return best_temp, weights


def apply_postprocessing(logits, temp, weights):
    scaled = logits / temp
    exp_l = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
    preds = np.argmax(probs * weights, axis=1)
    return probs, preds


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

    total_oof_logits = np.zeros((len(label_array), NUM_LABELS))
    total_test_logits = np.zeros((len(y_test), NUM_LABELS))
    best_overall_f1 = -1.0

    for seed in SEEDS:
        print("\n" + "=" * 65)
        print(f"TRAINING 5-FOLD INDOBERT + PARENT CONTEXT + R-DROP | SEED {seed}")
        print("=" * 65)
        set_seed(seed)
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        seed_oof_logits = np.zeros((len(label_array), NUM_LABELS))
        seed_test_logits = np.zeros((len(y_test), NUM_LABELS))

        for fold_num, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(label_array)), label_array), start=1):
            set_seed(seed + fold_num)
            train_loader = DataLoader(
                TensorDataset(input_ids[train_idx], attention_mask[train_idx], labels[train_idx]),
                batch_size=BATCH_SIZE,
                shuffle=True,
                pin_memory=USE_AMP,
            )
            val_loader = DataLoader(
                TensorDataset(input_ids[val_idx], attention_mask[val_idx]),
                batch_size=BATCH_SIZE,
                shuffle=False,
                pin_memory=USE_AMP,
            )

            model = IndoBERTClassifier(num_labels=NUM_LABELS, dropout_rate=0.2).to(DEVICE)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.04)
            optimizer = AdamW(get_smooth_llrd_params(model, ENCODER_LR, HEAD_LR, layer_decay=0.92, weight_decay=0.03))

            total_steps = len(train_loader) * MAX_EPOCHS
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
            )
            scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

            best_f1 = -1.0
            best_val_logits = None
            best_state = None

            for epoch in range(1, MAX_EPOCHS + 1):
                model.train()
                for ids_b, mask_b, lbl_b in train_loader:
                    ids_b, mask_b, lbl_b = ids_b.to(DEVICE), mask_b.to(DEVICE), lbl_b.to(DEVICE)
                    optimizer.zero_grad(set_to_none=True)

                    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                        logits1 = model(input_ids=ids_b, attention_mask=mask_b)
                        logits2 = model(input_ids=ids_b, attention_mask=mask_b)
                        loss = compute_rdrop_loss(logits1.float(), logits2.float(), lbl_b, criterion)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= old_scale:
                        scheduler.step()

                val_logits = predict_logits(model, val_loader)
                val_preds = np.argmax(val_logits, axis=1)
                val_acc = accuracy_score(label_array[val_idx], val_preds)
                val_f1 = f1_score(label_array[val_idx], val_preds, average="macro")
                print(f"Seed {seed} | Fold {fold_num} | Ep {epoch} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_val_logits = val_logits
                    best_state = {
                        k: (v.cpu().half() if v.is_floating_point() else v.cpu())
                        for k, v in model.state_dict().items()
                    }

            seed_oof_logits[val_idx] = best_val_logits
            if best_f1 > best_overall_f1:
                best_overall_f1 = best_f1
                torch.save(best_state, CHECKPOINT_DIR / "best_indobert_parent_rdrop.pt")

            model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
            seed_test_logits += predict_logits(model, test_loader) / N_SPLITS

            del model, optimizer, scaler, best_state
            if USE_AMP:
                torch.cuda.empty_cache()

        np.save(RESULTS_DIR / f"oof_logits_seed{seed}.npy", seed_oof_logits)
        np.save(RESULTS_DIR / f"test_logits_seed{seed}.npy", seed_test_logits)
        total_oof_logits += seed_oof_logits / len(SEEDS)
        total_test_logits += seed_test_logits / len(SEEDS)

    best_temp, best_w = optimize_postprocessing(total_oof_logits, label_array)
    oof_probs, oof_preds = apply_postprocessing(total_oof_logits, best_temp, best_w)

    print("\n" + "=" * 65)
    print(f"OOF REPORT (Temp={best_temp:.3f}, Class Weights={np.round(best_w, 4)})")
    print("=" * 65)
    print(f"OOF Accuracy: {accuracy_score(label_array, oof_preds):.4f} | OOF Macro F1: {f1_score(label_array, oof_preds, average='macro'):.4f}")
    print(classification_report(label_array, oof_preds, target_names=["Negatif", "Netral", "Positif"], digits=4))

    test_probs_raw, test_preds_raw = apply_postprocessing(total_test_logits, 1.0, np.array([1.0, 1.0, 1.0]))
    test_probs_opt, test_preds_opt = apply_postprocessing(total_test_logits, best_temp, best_w)

    print("=" * 65)
    print("GOLD-STANDARD TEST SET | 10-Model Parent-Context R-Drop (Raw)")
    print("=" * 65)
    print(f"Test Accuracy: {accuracy_score(y_test, test_preds_raw):.4f} | Test Macro F1: {f1_score(y_test, test_preds_raw, average='macro'):.4f}")
    print(classification_report(y_test, test_preds_raw, target_names=["Negatif", "Netral", "Positif"], digits=4))

    print("=" * 65)
    print("GOLD-STANDARD TEST SET | 10-Model Parent-Context R-Drop (+ Calibrated Threshold)")
    print("=" * 65)
    print(f"Test Accuracy: {accuracy_score(y_test, test_preds_opt):.4f} | Test Macro F1: {f1_score(y_test, test_preds_opt, average='macro'):.4f}")
    print(classification_report(y_test, test_preds_opt, target_names=["Negatif", "Netral", "Positif"], digits=4))

    cm = confusion_matrix(y_test, test_preds_opt)
    cm_df = pd.DataFrame(cm, index=["True_Negatif", "True_Netral", "True_Positif"], columns=["Pred_Negatif", "Pred_Netral", "Pred_Positif"])
    print("Confusion Matrix (Test Set):")
    print(cm_df)

    cm_df.to_csv(RESULTS_DIR / "test_confusion_matrix.csv")
    pd.DataFrame({
        "comment_id": test_data["comment_ids"],
        "true_label": [ID2LABEL[i] for i in y_test],
        "pred_label_raw": [ID2LABEL[i] for i in test_preds_raw],
        "pred_label_opt": [ID2LABEL[i] for i in test_preds_opt],
        "prob_Negatif": test_probs_opt[:, 0],
        "prob_Netral": test_probs_opt[:, 1],
        "prob_Positif": test_probs_opt[:, 2],
    }).to_csv(RESULTS_DIR / "test_parent_rdrop_predictions.csv", index=False)


if __name__ == "__main__":
    main()
