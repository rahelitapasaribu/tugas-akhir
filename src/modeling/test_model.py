"""
Evaluasi Akhir Fine-Tuned IndoBERT (5-Fold Soft-Voting Ensemble + Optimal Threshold)
pada Data Uji (test.csv - 1.500 komentar).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.optimize import minimize
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
TOKENIZED_DIR = PROJECT_ROOT / "data" / "processed" / "tokenized_audited_v3_large"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics_audited_v3_norm"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "indobert_base_5fold"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
N_SPLITS = 5
BATCH_SIZE = 32


def get_optimal_weights(oof_path: Path):
    df_oof = pd.read_csv(oof_path)
    label2id = {"Negatif": 0, "Netral": 1, "Positif": 2}
    y_true = df_oof["true_label"].map(label2id).values
    probs = df_oof[["prob_Negatif", "prob_Netral", "prob_Positif"]].values

    def loss_func(weights):
        preds = np.argmax(probs * weights, axis=1)
        return -(0.5 * accuracy_score(y_true, preds) + 0.5 * f1_score(y_true, preds, average="macro"))

    res = minimize(loss_func, [1.0, 1.0, 1.0], method="Nelder-Mead", options={"maxiter": 500})
    best_weights = res.x / res.x[0]
    y_pred_oof = np.argmax(probs * best_weights, axis=1)
    print(f"Optimal OOF Class Weights [Neg, Net, Pos]: {np.round(best_weights, 4)}")
    print(f"Tuned OOF Accuracy: {accuracy_score(y_true, y_pred_oof):.4f} | Tuned OOF Macro F1: {f1_score(y_true, y_pred_oof, average='macro'):.4f}\n")
    return best_weights


def predict_fold(model_path: Path, loader: DataLoader) -> np.ndarray:
    model = IndoBERTClassifier(num_labels=3, dropout_rate=0.2).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    fold_probs = []
    with torch.no_grad():
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits.float(), dim=1)
            fold_probs.append(probs.cpu().numpy())

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return np.vstack(fold_probs)


def main():
    best_weights = get_optimal_weights(RESULTS_DIR / "oof_predictions.csv")

    test_data = torch.load(TOKENIZED_DIR / "test.pt", map_location="cpu", weights_only=True)
    test_dataset = TensorDataset(test_data["input_ids"], test_data["attention_mask"])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    y_true = test_data["labels"].numpy()
    comment_ids = test_data["comment_ids"]

    ensemble_probs = np.zeros((len(y_true), 3))
    for fold_num in range(1, N_SPLITS + 1):
        ckpt_path = CHECKPOINT_DIR / f"best_fold_{fold_num}.pt"
        probs_f = predict_fold(ckpt_path, test_loader)
        preds_f = np.argmax(probs_f, axis=1)
        print(f"Fold {fold_num} Test Accuracy: {accuracy_score(y_true, preds_f):.4f} | Macro F1: {f1_score(y_true, preds_f, average='macro'):.4f}")
        ensemble_probs += probs_f / N_SPLITS

    # 1. Raw 5-Fold Soft-Voting
    preds_raw = np.argmax(ensemble_probs, axis=1)
    print("\n" + "=" * 60)
    print("HASIL EVALUASI AKHIR TEST SET (5-Fold IndoBERT Soft-Voting)")
    print("=" * 60)
    print(f"Test Accuracy       : {accuracy_score(y_true, preds_raw):.4f}")
    print(f"Test Precision Macro: {precision_score(y_true, preds_raw, average='macro'):.4f}")
    print(f"Test Recall Macro   : {recall_score(y_true, preds_raw, average='macro'):.4f}")
    print(f"Test F1-Score Macro : {f1_score(y_true, preds_raw, average='macro'):.4f}\n")
    print(classification_report(y_true, preds_raw, target_names=["Negatif", "Netral", "Positif"], digits=4))

    # 2. 5-Fold Soft-Voting + Optimal Threshold Tuning
    preds_tuned = np.argmax(ensemble_probs * best_weights, axis=1)
    print("=" * 60)
    print("HASIL EVALUASI AKHIR TEST SET (+ Optimal Class Threshold)")
    print("=" * 60)
    print(f"Test Accuracy       : {accuracy_score(y_true, preds_tuned):.4f}")
    print(f"Test Precision Macro: {precision_score(y_true, preds_tuned, average='macro'):.4f}")
    print(f"Test Recall Macro   : {recall_score(y_true, preds_tuned, average='macro'):.4f}")
    print(f"Test F1-Score Macro : {f1_score(y_true, preds_tuned, average='macro'):.4f}\n")
    print(classification_report(y_true, preds_tuned, target_names=["Negatif", "Netral", "Positif"], digits=4))

    cm = confusion_matrix(y_true, preds_tuned)
    cm_df = pd.DataFrame(cm, index=["True_Negatif", "True_Netral", "True_Positif"], columns=["Pred_Negatif", "Pred_Netral", "Pred_Positif"])
    print("Confusion Matrix (Test Set - 1.500 Data):")
    print(cm_df)

    cm_df.to_csv(RESULTS_DIR / "test_confusion_matrix.csv")
    pd.DataFrame({
        "comment_id": comment_ids,
        "true_label": [ID2LABEL[i] for i in y_true],
        "pred_label_raw": [ID2LABEL[i] for i in preds_raw],
        "pred_label_tuned": [ID2LABEL[i] for i in preds_tuned],
        "prob_Negatif": ensemble_probs[:, 0],
        "prob_Netral": ensemble_probs[:, 1],
        "prob_Positif": ensemble_probs[:, 2],
    }).to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    print("\nSeluruh hasil evaluasi akhir disimpan di:", RESULTS_DIR)


if __name__ == "__main__":
    main()
