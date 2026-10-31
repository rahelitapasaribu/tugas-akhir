# Menggabungkan 79 label final dari pilot_review_queue.csv ke dataset pilot 300 komentar tanpa mengubah file asli.

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

PILOT_FILE = BASE_DIR / "data" / "processed" / "sentiment_pilot.csv"
REVIEW_FILE = BASE_DIR / "data" / "processed" / "pilot_review_queue.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "sentiment_pilot_final.csv"

VALID_LABELS = {"Positif", "Negatif", "Netral"}

# Membaca dataset
pilot = pd.read_csv(PILOT_FILE, dtype={"comment_id": str})
review = pd.read_csv(REVIEW_FILE, dtype={"comment_id": str})

# Memeriksa kolom wajib
required_pilot = {"comment_id", "manual_sentiment"}
required_review = {"comment_id", "final_sentiment"}

if not required_pilot.issubset(pilot.columns):
    raise ValueError(f"Kolom dataset pilot tidak lengkap: {required_pilot - set(pilot.columns)}")

if not required_review.issubset(review.columns):
    raise ValueError(f"Kolom review queue tidak lengkap: {required_review - set(review.columns)}")

# Memeriksa ID komentar duplikat
if pilot["comment_id"].duplicated().any():
    raise ValueError("Ada comment_id duplikat di sentiment_pilot.csv.")

if review["comment_id"].duplicated().any():
    raise ValueError("Ada comment_id duplikat di pilot_review_queue.csv.")

# Memeriksa apakah seluruh label final sudah diisi
review["final_sentiment"] = review["final_sentiment"].astype("string").str.strip().str.capitalize()

if review["final_sentiment"].isna().any() or (review["final_sentiment"] == "").any():
    raise ValueError("Masih ada final_sentiment yang kosong.")

invalid_labels = set(review["final_sentiment"].dropna()) - VALID_LABELS

if invalid_labels:
    raise ValueError(f"Ada label tidak valid: {invalid_labels}")

# Memeriksa apakah semua ID review ditemukan di dataset pilot
missing_ids = set(review["comment_id"]) - set(pilot["comment_id"])

if missing_ids:
    raise ValueError(f"Ada {len(missing_ids)} comment_id review yang tidak ditemukan di dataset pilot.")

# Menggabungkan label final berdasarkan comment_id
label_map = review.set_index("comment_id")["final_sentiment"]

pilot["sentiment_final"] = pilot["manual_sentiment"].astype("string").str.strip().str.capitalize()

matched = pilot["comment_id"].isin(label_map.index)
pilot.loc[matched, "sentiment_final"] = pilot.loc[matched, "comment_id"].map(label_map)

# Memeriksa hasil akhir
if pilot["sentiment_final"].isna().any() or (pilot["sentiment_final"] == "").any():
    raise ValueError("Masih ada label sentiment_final yang kosong.")

invalid_final = set(pilot["sentiment_final"].dropna()) - VALID_LABELS

if invalid_final:
    raise ValueError(f"Ada label final yang tidak valid: {invalid_final}")

# Menyimpan dataset baru
pilot.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("=== PENGGABUNGAN LABEL SELESAI ===")
print(f"Total komentar       : {len(pilot)}")
print(f"Komentar diperbarui  : {matched.sum()}")
print(f"Komentar tetap manual: {(~matched).sum()}")
print("\nDistribusi sentiment_final:")
print(pilot["sentiment_final"].value_counts())
print(f"\nFile tersimpan di: {OUTPUT_FILE}")