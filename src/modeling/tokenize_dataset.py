"""
Melakukan tokenisasi dataset menggunakan tokenizer IndoBERT.

Input:
data/processed/splits/train.csv
data/processed/splits/validation.csv
data/processed/splits/test.csv

Output:
data/processed/tokenized/
"""

from pathlib import Path
import json

import pandas as pd
import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "tokenized"

MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LENGTH = 128

TEXT_COLUMN = "comment_preprocessed"
LABEL_COLUMN = "label_final"
ID_COLUMN = "comment_id"

LABEL2ID = {
    "Negatif": 0,
    "Netral": 1,
    "Positif": 2,
}

SPLITS = ["train", "validation", "test"]


def main():
    """Memuat tokenizer dan melakukan tokenisasi seluruh subset."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for split in SPLITS:
        input_path = INPUT_DIR / f"{split}.csv"

        if not input_path.exists():
            raise FileNotFoundError(
                f"File tidak ditemukan: {input_path}"
            )

        df = pd.read_csv(input_path)

        required_columns = [
            ID_COLUMN,
            TEXT_COLUMN,
            LABEL_COLUMN,
        ]

        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Kolom tidak ditemukan pada {split}: {missing}"
            )

        if df[TEXT_COLUMN].isna().any():
            raise ValueError(
                f"Terdapat teks kosong pada {split}"
            )

        if not set(df[LABEL_COLUMN].unique()).issubset(LABEL2ID):
            raise ValueError(
                f"Label tidak valid pada {split}"
            )

        encoded = tokenizer(
            df[TEXT_COLUMN].astype(str).tolist(),
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        labels = df[LABEL_COLUMN].map(LABEL2ID).tolist()

        tokenized_data = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
            "comment_ids": df[ID_COLUMN].astype(str).tolist(),
        }

        output_path = OUTPUT_DIR / f"{split}.pt"
        torch.save(tokenized_data, output_path)

        print(f"{split}: {len(df)} data berhasil ditokenisasi.")
        print(f"Disimpan di: {output_path}")

    tokenizer.save_pretrained(OUTPUT_DIR / "tokenizer")

    label_mapping = {
        "label2id": LABEL2ID,
        "id2label": {str(v): k for k, v in LABEL2ID.items()},
    }

    with open(OUTPUT_DIR / "label_mapping.json", "w") as file:
        json.dump(label_mapping, file, indent=4)

    print("\nTokenisasi seluruh dataset berhasil.")
    print(f"Max length: {MAX_LENGTH}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
