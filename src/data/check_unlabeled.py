# Memeriksa komentar yang belum mendapatkan label sentimen dari Qwen

import pandas as pd

INPUT_FILE = "data/processed/comments_labeled.csv"

df = pd.read_csv(INPUT_FILE, dtype={"comment_id": str})

unlabeled = df[
    df["sentiment_qwen"].isna()
    | ~df["sentiment_qwen"].isin(["Positif", "Negatif", "Netral"])
]

print("Jumlah komentar belum terlabel:", len(unlabeled))
print(unlabeled[["comment_id", "comment", "sentiment_qwen", "reason_qwen"]].to_string(index=False))

unlabeled.to_csv(
    "data/processed/comments_unlabeled.csv",
    index=False,
    encoding="utf-8-sig"
)