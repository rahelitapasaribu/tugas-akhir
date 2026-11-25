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
import re
import pandas as pd
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "tokenized"

MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LENGTH = 160  # Dinaikkan dari 128 agar judul + komentar panjang tidak terpotong

TITLE_COLUMN = "video_title"
TEXT_COLUMN = "comment_original"  # Gunakan teks natural (cased) agar emosi dari kapitalisasi/emoji terjaga
LABEL_COLUMN = "label_final"
ID_COLUMN = "comment_id"

LABEL2ID = {"Negatif": 0, "Netral": 1, "Positif": 2}
SPLITS = ["train", "validation", "test"]

def clean_minimal(text: str) -> str:
    # Hapus karakter tak terlihat (zero-width space) dan spasi/newline berlebih saja
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for split in SPLITS:
        df = pd.read_csv(INPUT_DIR / f"{split}.csv")

        titles = df[TITLE_COLUMN].fillna("").apply(clean_minimal).tolist()
        comments = df[TEXT_COLUMN].fillna("").apply(clean_minimal).tolist()

        encoded = tokenizer(
            titles,
            comments,
            padding="max_length",
            truncation="only_second",  # Hanya potong komentar jika melebihi max_length
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

        torch.save(tokenized_data, OUTPUT_DIR / f"{split}.pt")
        print(f"{split}: {len(df)} data berhasil ditokenisasi.")

    tokenizer.save_pretrained(OUTPUT_DIR / "tokenizer")
    with open(OUTPUT_DIR / "label_mapping.json", "w") as f:
        json.dump({"label2id": LABEL2ID, "id2label": {str(v): k for k, v in LABEL2ID.items()}}, f, indent=4)

if __name__ == "__main__":
    main()
