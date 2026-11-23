
"""
Fine-tuning IndoBERT menggunakan custom PyTorch training loop
dan 5-Fold Stratified Cross-Validation.

Data:
- Gabungan training dan validation: 8.500 data
- Test set: tidak digunakan dalam proses tuning

Hyperparameter:
- Learning rate: 2e-5, 3e-5
- Batch size: 16, 32
- Epoch maksimum: 5
- Early stopping patience: 2
"""
from pathlib import Path
import random
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from model import IndoBERTClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKENIZED_DIR = PROJECT_ROOT / "data" / "processed" / "tokenized"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
MAX_EPOCHS = 5
PATIENCE = 2
LEARNING_RATES = [2e-5, 3e-5]
BATCH_SIZES = [16, 32]
NUM_LABELS = 3
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
USE_AMP = DEVICE.type == "cuda"

def set_seed(seed=RANDOM_STATE):
    """Mengatur random seed untuk reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_tokenized_data():
    """Memuat dan menggabungkan tokenisasi training dan validation."""

    train_data = torch.load(
        TOKENIZED_DIR / "train.pt",
        map_location="cpu",
        weights_only=True,
    )

    validation_data = torch.load(
        TOKENIZED_DIR / "validation.pt",
        map_location="cpu",
        weights_only=True,
    )

    input_ids = torch.cat(
        [
            train_data["input_ids"],
            validation_data["input_ids"],
        ],
        dim=0,
    )

    attention_mask = torch.cat(
        [
            train_data["attention_mask"],
            validation_data["attention_mask"],
        ],
        dim=0,
    )

    labels = torch.cat(
        [
            train_data["labels"],
            validation_data["labels"],
        ],
        dim=0,
    )

    print("Data training + validation berhasil digabung.")
    print("Jumlah data:", len(labels))

    return input_ids, attention_mask, labels

def create_loader(input_ids, attention_mask, labels, batch_size, shuffle):
    """Membuat DataLoader dari indeks subset."""

    dataset = TensorDataset(
        input_ids,
        attention_mask,
        labels,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )

def evaluate(model, loader, criterion):
    """Menghitung loss dan metrik klasifikasi."""

    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:

            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP,
            ):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                loss = criterion(logits.float(), labels)

            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    metrics = {
        "val_loss": total_loss / len(loader),
        "accuracy": accuracy_score(
            all_labels, all_predictions
        ),
        "precision_macro": precision_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        ),
    }

    return metrics

def train_one_fold(
    input_ids,
    attention_mask,
    labels,
    train_indices,
    val_indices,
    learning_rate,
    batch_size,
    fold_number,
):
    """Melatih dan mengevaluasi model pada satu fold."""

    set_seed(RANDOM_STATE + fold_number)

    train_loader = create_loader(
        input_ids[train_indices],
        attention_mask[train_indices],
        labels[train_indices],
        batch_size,
        shuffle=True,
    )

    val_loader = create_loader(
        input_ids[val_indices],
        attention_mask[val_indices],
        labels[val_indices],
        batch_size,
        shuffle=False,
    )

    model = IndoBERTClassifier(
        num_labels=NUM_LABELS
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP,
    )

    best_f1 = -1.0
    best_metrics = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()
        total_train_loss = 0.0

        for batch in train_loader:

            input_ids_batch = batch[0].to(DEVICE)
            attention_mask_batch = batch[1].to(DEVICE)
            labels_batch = batch[2].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP,
            ):
                logits = model(
                    input_ids=input_ids_batch,
                    attention_mask=attention_mask_batch,
                )

                loss = criterion(
                    logits.float(),
                    labels_batch,
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
        )

        print(
            f"Fold {fold_number} | "
            f"Epoch {epoch}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val F1: {val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_f1:

            best_f1 = val_metrics["f1_macro"]
            best_metrics = val_metrics.copy()
            best_epoch = epoch
            patience_counter = 0

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping dijalankan.")
            break

    del model
    del optimizer
    del scaler

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "best_epoch": best_epoch,
        **best_metrics,
    }

def main():
    set_seed()

    print("Device:", DEVICE)
    print("Mixed precision:", USE_AMP)

    input_ids, attention_mask, labels = load_tokenized_data()
    label_array = labels.numpy()
    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    experiments = list(
        itertools.product(
            LEARNING_RATES,
            BATCH_SIZES,
        )
    )

    fold_results = []
    experiment_results = []

    for experiment_number, (lr, batch_size) in enumerate(
        experiments, start=1
    ):

        print("\n" + "=" * 60)
        print(
            f"EXPERIMENT {experiment_number}/{len(experiments)}"
        )
        print(f"Learning rate: {lr}")
        print(f"Batch size: {batch_size}")
        print("=" * 60)

        current_results = []

        for fold_number, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(label_array)), label_array),
            start=1,
        ):

            print(f"\nMemulai Fold {fold_number}/{N_SPLITS}")

            result = train_one_fold(
                input_ids,
                attention_mask,
                labels,
                train_idx,
                val_idx,
                lr,
                batch_size,
                fold_number,
            )

            result["experiment"] = experiment_number
            result["fold"] = fold_number
            result["learning_rate"] = lr
            result["batch_size"] = batch_size

            fold_results.append(result)
            current_results.append(result)

            pd.DataFrame(fold_results).to_csv(
                RESULTS_DIR / "cv_results.csv",
                index=False,
            )

        current_df = pd.DataFrame(current_results)

        summary = {
            "experiment": experiment_number,
            "learning_rate": lr,
            "batch_size": batch_size,
            "mean_accuracy": current_df["accuracy"].mean(),
            "std_accuracy": current_df["accuracy"].std(),
            "mean_precision_macro": current_df[
                "precision_macro"
            ].mean(),
            "mean_recall_macro": current_df[
                "recall_macro"
            ].mean(),
            "mean_f1_macro": current_df["f1_macro"].mean(),
            "std_f1_macro": current_df["f1_macro"].std(),
            "mean_best_epoch": current_df["best_epoch"].mean(),
        }

        experiment_results.append(summary)

        pd.DataFrame(experiment_results).to_csv(
            RESULTS_DIR / "hyperparameter_results.csv",
            index=False,
        )

        print("\nRingkasan eksperimen:")
        print(pd.Series(summary))

    summary_df = pd.DataFrame(experiment_results)

    summary_df = summary_df.sort_values(
        by="mean_f1_macro",
        ascending=False,
    )

    summary_df.to_csv(
        RESULTS_DIR / "hyperparameter_results.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("HASIL HYPERPARAMETER TUNING")
    print("=" * 60)

    print(summary_df.to_string(index=False))

    print("\nKonfigurasi terbaik:")
    print(summary_df.iloc[0].to_string())

    print("\nCross-validation selesai.")
    print("Hasil disimpan di:", RESULTS_DIR)

if __name__ == "__main__":
    main()
