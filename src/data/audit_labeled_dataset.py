
# Mengaudit kelengkapan dan validitas label sentimen sebelum sampling dataset.

import pandas as pd

INPUT_FILE = "data/processed/comments_labeled.csv"
OUTPUT_FILE = "data/processed/comments_audit_issues.csv"

VALID_LABELS = {"Positif", "Negatif", "Netral"}
EXPECTED_ROWS = 47136

print("Membaca dataset...")

df = pd.read_csv(
    INPUT_FILE,
    dtype={"comment_id": str},
    low_memory=False
)

# Memastikan kolom yang diperlukan tersedia
required_columns = {
    "comment_id", "comment", "sentiment_qwen", "reason_qwen"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Kolom tidak ditemukan: {missing_columns}")

# Pemeriksaan identitas data
total_rows = len(df)
missing_id = df["comment_id"].isna()
duplicate_id = df["comment_id"].duplicated(keep=False)

# Pemeriksaan label
labels = df["sentiment_qwen"].fillna("").astype(str).str.strip()

missing_label = labels.eq("")
invalid_label = ~labels.isin(VALID_LABELS) & ~missing_label

# Pemeriksaan teks dan alasan
empty_comment = (
    df["comment"].fillna("").astype(str).str.strip().eq("")
)

empty_reason = (
    df["reason_qwen"].fillna("").astype(str).str.strip().eq("")
)

# Menampilkan hasil audit
print("\n========== AUDIT DATASET ==========")
print(f"Total data             : {total_rows}")
print(f"Comment ID unik        : {df['comment_id'].nunique()}")
print(f"Comment ID kosong      : {missing_id.sum()}")
print(f"Baris dengan ID duplikat: {duplicate_id.sum()}")
print(f"Label kosong           : {missing_label.sum()}")
print(f"Label tidak valid      : {invalid_label.sum()}")
print(f"Komentar kosong        : {empty_comment.sum()}")
print(f"Alasan label kosong    : {empty_reason.sum()}")

print("\n========== DISTRIBUSI SENTIMEN ==========")
print(labels[labels.isin(VALID_LABELS)].value_counts())

# Status pemeriksaan
checks = {
    "Jumlah data sesuai 47.136": total_rows == EXPECTED_ROWS,
    "Semua ID terisi": missing_id.sum() == 0,
    "Semua ID unik": duplicate_id.sum() == 0,
    "Semua label terisi": missing_label.sum() == 0,
    "Semua label valid": invalid_label.sum() == 0,
}

print("\n========== STATUS AUDIT ==========")

for name, passed in checks.items():
    print(f"{name}: {'LULUS' if passed else 'PERLU DIPERIKSA'}")

# Menyimpan data yang perlu diperiksa
problem_mask = (
    missing_id
    | duplicate_id
    | missing_label
    | invalid_label
    | empty_comment
)

issues = df.loc[problem_mask].copy()

issues.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nJumlah baris perlu diperiksa: {len(issues)}")
print(f"File pemeriksaan: {OUTPUT_FILE}")
print("====================================")