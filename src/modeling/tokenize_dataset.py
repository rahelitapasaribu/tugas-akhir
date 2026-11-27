from pathlib import Path
import json
import re
import pandas as pd
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "tokenized_audited_v3_clean"

MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LENGTH = 224

TITLE_COLUMN = "video_title"
TEXT_COLUMN = "comment_original"
LABEL_COLUMN = "label_final"
ID_COLUMN = "comment_id"

LABEL2ID = {"Negatif": 0, "Netral": 1, "Positif": 2}
SPLIT_FILES = {
    "train": "train_audited_v3.csv",
    "validation": "validation_audited_v3.csv",
    "test": "test_audited_final.csv",
}


def clean_text(text: str) -> str:
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", str(text))
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_context(row) -> str:
    title = clean_text(row[TITLE_COLUMN]) if pd.notna(row[TITLE_COLUMN]) else ""
    reply_tag = "Balasan" if str(row.get("is_reply", False)).lower() == "true" else "Komentar Utama"
    return f"{reply_tag} | {title}"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for split, filename in SPLIT_FILES.items():
        df = pd.read_csv(INPUT_DIR / filename)
        contexts = df.apply(build_context, axis=1).tolist()
        comments = df[TEXT_COLUMN].fillna("").apply(clean_text).tolist()

        encoded = tokenizer(
            contexts,
            comments,
            padding="max_length",
            truncation="only_second",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        labels = df[LABEL_COLUMN].map(LABEL2ID).tolist()
        torch.save(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": torch.tensor(labels, dtype=torch.long),
                "comment_ids": df[ID_COLUMN].astype(str).tolist(),
            },
            OUTPUT_DIR / f"{split}.pt",
        )
        print(f"{split} ({filename}): {len(df)} data berhasil ditokenisasi (max_length={MAX_LENGTH}).")

    tokenizer.save_pretrained(OUTPUT_DIR / "tokenizer")
    with open(OUTPUT_DIR / "label_mapping.json", "w") as f:
        json.dump({"label2id": LABEL2ID, "id2label": {str(v): k for k, v in LABEL2ID.items()}}, f, indent=4)


if __name__ == "__main__":
    main()
