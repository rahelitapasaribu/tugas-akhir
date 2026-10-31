# Memeriksa kualitas dataset pilot final, termasuk duplikasi, data kosong, validitas label, dan distribusi sentimen tanpa menghapus komentar.

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data" / "processed" / "sentiment_pilot_final.csv"

df = pd.read_csv(INPUT_FILE, dtype={"comment_id": str})

print("=== AUDIT DATASET PILOT FINAL ===")
print(f"Total baris data: {len(df)}")

# Memeriksa kolom
print("\nKolom yang tersedia:")
print(df.columns.tolist())

# Memeriksa duplikasi ID
print("\nDuplikasi comment_id:")
print(df["comment_id"].duplicated().sum())

# Memeriksa duplikasi teks komentar jika kolom tersedia
comment_column = next(
    (col for col in ["comment_text", "comment", "text"] if col in df.columns),
    None
)

if comment_column:
    print("\nDuplikasi teks komentar:")
    print(df[comment_column].duplicated().sum())

    print("\nKomentar kosong:")
    print(df[comment_column].isna().sum() + (df[comment_column].astype("string").str.strip() == "").sum())

# Memeriksa label
print("\nData sentiment_final kosong:")
print(df["sentiment_final"].isna().sum())

print("\nDistribusi sentiment_final:")
print(df["sentiment_final"].value_counts(dropna=False))

valid_labels = {"Positif", "Negatif", "Netral"}
invalid = set(df["sentiment_final"].dropna().unique()) - valid_labels

print("\nLabel tidak valid:")
print(invalid if invalid else "Tidak ada")

# Ringkasan
print("\n=== KESIMPULAN PEMERIKSAAN ===")
print(f"Jumlah data sesuai 300: {len(df) == 300}")
print(f"ID unik: {df['comment_id'].nunique() == len(df)}")
print(f"Semua label terisi: {df['sentiment_final'].notna().all()}")
print(f"Semua label valid: {len(invalid) == 0}")