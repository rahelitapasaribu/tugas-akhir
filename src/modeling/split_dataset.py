
"""
Membagi dataset sentimen menjadi training, validation, dan testing.

Metode:
- Stratified train-validation-test split
- Proporsi 70:15:15
- Random state 42

Input:
data/processed/comments_labeled_final.csv

Output:
data/processed/splits/train.csv
data/processed/splits/validation.csv
data/processed/splits/test.csv
results/metrics/split_distribution.csv
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "comments_labeled_final.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"

METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

TEXT_COLUMN = "comment_preprocessed"
LABEL_COLUMN = "label_final"
ID_COLUMN = "comment_id"

RANDOM_STATE = 42

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15


def main():
    """Memvalidasi, membagi, dan menyimpan dataset."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        ID_COLUMN,
        TEXT_COLUMN,
        LABEL_COLUMN,
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Kolom tidak ditemukan: {missing_columns}")

    if df[required_columns].isna().any().any():
        raise ValueError("Terdapat nilai kosong pada kolom utama.")

    if df[ID_COLUMN].duplicated().any():
        raise ValueError("Terdapat comment_id duplikat.")

    valid_labels = {"Positif", "Negatif", "Netral"}

    if set(df[LABEL_COLUMN].unique()) != valid_labels:
        raise ValueError("Label dataset tidak sesuai dengan tiga kelas.")

    # Membagi training dan data sementara
    train_df, temp_df = train_test_split(
        df,
        test_size=TEST_SIZE + VALIDATION_SIZE,
        stratify=df[LABEL_COLUMN],
        random_state=RANDOM_STATE,
    )

    # Membagi data sementara menjadi validation dan testing
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df[LABEL_COLUMN],
        random_state=RANDOM_STATE,
    )

    splits = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Menyimpan setiap subset
    for name, subset in splits.items():
        output_path = OUTPUT_DIR / f"{name}.csv"

        subset.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"{name}: {len(subset)} data")
        print(output_path)

    # Membuat laporan distribusi label
    distribution = []

    for split_name, subset in splits.items():
        counts = subset[LABEL_COLUMN].value_counts()

        for label in sorted(valid_labels):
            count = int(counts.get(label, 0))

            distribution.append({
                "split": split_name,
                "label": label,
                "jumlah": count,
                "persentase": round(
                    count / len(subset) * 100, 2
                ),
            })

    distribution_df = pd.DataFrame(distribution)

    distribution_path = (
        METRICS_DIR / "split_distribution.csv"
    )

    distribution_df.to_csv(
        distribution_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Memastikan seluruh data telah terbagi tanpa kehilangan data
    total_split = sum(len(subset) for subset in splits.values())

    assert total_split == len(df)
    assert len(train_df) == 7000
    assert len(validation_df) == 1500
    assert len(test_df) == 1500

    print("\nDistribusi label:")
    print(
        distribution_df.pivot(
            index="label",
            columns="split",
            values="jumlah",
        )
    )

    print("\nFinal split berhasil.")
    print("Total data:", total_split)
    print("Laporan:", distribution_path)


if __name__ == "__main__":
    main()
