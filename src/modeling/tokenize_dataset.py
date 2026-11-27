"""
Tokenisasi dataset hasil audit v3 menggunakan kamus normalisasi terkurasi
(Subbab 7.3.4) untuk Fine-Tuning IndoBERT.
"""
from pathlib import Path
import json
import re
import pandas as pd
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
VOCAB_DIR = PROJECT_ROOT / "data" / "processed" / "vocabulary_audit"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "tokenized_audited_v3_norm"

MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LENGTH = 160

TITLE_COLUMN = "video_title"
TEXT_COLUMN = "comment_original"
LABEL_COLUMN = "label_final"
ID_COLUMN = "comment_id"

LABEL2ID = {"Negatif": 0, "Netral": 1, "Positif": 2}
SPLIT_FILES = {
    "train": "train_audited_v3.csv",
    "validation": "validation_audited_v3.csv",
    "test": "test.csv",
}


def load_normalization_dict() -> dict:
    dict_paths = [
        VOCAB_DIR / "normalization_dictionary_final.csv",
        VOCAB_DIR / "kamusalay_extension" / "normalization_dictionary_final.csv",
    ]
    merged_map = {}
    for path in dict_paths:
        if path.exists():
            df_d = pd.read_csv(path).dropna(subset=["word", "normalisasi"])
            for _, row in df_d.iterrows():
                w = str(row["word"]).strip().lower()
                norm = str(row["normalisasi"]).strip().lower()
                if w and norm and w != norm:
                    merged_map[w] = norm
    print(f"Kamus normalisasi dimuat: {len(merged_map)} entri unik.")
    return merged_map


SLANG_DICT = load_normalization_dict()
SLANG_PATTERN = (
    re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(SLANG_DICT.keys(), key=len, reverse=True)) + r")\b", re.IGNORECASE)
    if SLANG_DICT else None
)


def _replace_slang(match):
    word = match.group(0)
    replacement = SLANG_DICT[word.lower()]
    if word.isupper() and len(word) > 1:
        return replacement.upper()
    if word[0].isupper():
        return replacement.capitalize()
    return replacement


def normalize_text(text: str) -> str:
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", str(text))
    text = re.sub(r"http\S+|www\.\S+", "", text)
    if SLANG_PATTERN is not None:
        text = SLANG_PATTERN.sub(_replace_slang, text)
    text = re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", text)
    if SLANG_PATTERN is not None:
        text = SLANG_PATTERN.sub(_replace_slang, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_context(row) -> str:
    title = normalize_text(row[TITLE_COLUMN]) if pd.notna(row[TITLE_COLUMN]) else ""
    reply_tag = "Balasan" if str(row.get("is_reply", False)).lower() == "true" else "Komentar Utama"
    return f"{reply_tag} | {title}"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for split, filename in SPLIT_FILES.items():
        df = pd.read_csv(INPUT_DIR / filename)

        contexts = df.apply(build_context, axis=1).tolist()
        comments = df[TEXT_COLUMN].fillna("").apply(normalize_text).tolist()

        encoded = tokenizer(
            contexts,
            comments,
            padding="max_length",
            truncation="only_second",
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
        print(f"{split} ({filename}): {len(df)} data berhasil ditokenisasi.")

    tokenizer.save_pretrained(OUTPUT_DIR / "tokenizer")
    with open(OUTPUT_DIR / "label_mapping.json", "w") as f:
        json.dump({"label2id": LABEL2ID, "id2label": {str(v): k for k, v in LABEL2ID.items()}}, f, indent=4)


if __name__ == "__main__":
    main()
