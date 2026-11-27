"""
Tokenisasi dataset audit v3 dengan injeksi konteks komentar induk (parent_id)
dari comments_raw.csv untuk meningkatkan akurasi pada komentar balasan.
"""
from pathlib import Path
import html
import json
import re
import pandas as pd
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "comments_raw.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
AUDIT_DIR = PROJECT_ROOT / "data" / "processed" / "label_audit"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "tokenized_v3_parent_ctx"

MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LENGTH = 256

LABEL2ID = {"Negatif": 0, "Netral": 1, "Positif": 2}


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = html.unescape(str(text))
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)
    text = re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_parent_lookup() -> dict:
    df_raw = pd.read_csv(RAW_PATH, usecols=["comment_id", "comment"])
    df_raw = df_raw.drop_duplicates("comment_id")
    return {
        str(cid): clean_text(cmt)[:140]
        for cid, cmt in zip(df_raw["comment_id"], df_raw["comment"])
    }


def build_context(row, parent_map: dict) -> str:
    title = clean_text(row.get("video_title", ""))
    is_rep = str(row.get("is_reply", False)).lower() == "true"
    if is_rep:
        pid = str(row.get("parent_id", ""))
        parent_txt = parent_map.get(pid, "")
        if parent_txt:
            return f"Balasan terhadap: {parent_txt} | Video: {title}"
        return f"Balasan | Video: {title}"
    return f"Komentar Utama | Video: {title}"


def load_split_df(split: str) -> pd.DataFrame:
    if split == "train":
        return pd.read_csv(SPLITS_DIR / "train_audited_v3.csv")
    if split == "validation":
        return pd.read_csv(SPLITS_DIR / "validation_audited_v3.csv")

    # Gabungkan splits/test.csv (memiliki parent_id) dengan label_audited final
    df_test = pd.read_csv(SPLITS_DIR / "test.csv")
    audit_file = AUDIT_DIR / "test_audited_final.csv"
    if not audit_file.exists():
        audit_file = SPLITS_DIR / "test_audited_final.csv"
    if audit_file.exists():
        df_aud = pd.read_csv(audit_file)[["comment_id", "label_audited"]].drop_duplicates("comment_id")
        df_test = df_test.merge(df_aud, on="comment_id", how="left")
        df_test["label_final"] = df_test["label_audited"].fillna(df_test["label_final"])
    return df_test


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    parent_map = load_parent_lookup()
    print(f"Lookup komentar induk dimuat: {len(parent_map)} entri.")

    for split in ["train", "validation", "test"]:
        df = load_split_df(split)
        contexts = [build_context(row, parent_map) for _, row in df.iterrows()]
        comments = df["comment_original"].fillna("").apply(clean_text).tolist()

        encoded = tokenizer(
            contexts,
            comments,
            padding="max_length",
            truncation="only_second",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        labels = df["label_final"].map(LABEL2ID).tolist()
        torch.save(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": torch.tensor(labels, dtype=torch.long),
                "comment_ids": df["comment_id"].astype(str).tolist(),
            },
            OUTPUT_DIR / f"{split}.pt",
        )
        matched = sum(1 for c in contexts if c.startswith("Balasan terhadap:"))
        print(f"{split}: {len(df)} data ditokenisasi (balasan dengan konteks induk: {matched}).")

    tokenizer.save_pretrained(OUTPUT_DIR / "tokenizer")
    with open(OUTPUT_DIR / "label_mapping.json", "w") as f:
        json.dump({"label2id": LABEL2ID, "id2label": {str(v): k for k, v in LABEL2ID.items()}}, f, indent=4)


if __name__ == "__main__":
    main()
