# Menganalisis perbedaan label Ollama dan label manual pada data pilot untuk menemukan pola kesalahan anotasi.

import pandas as pd
from pathlib import Path

# Lokasi file
INPUT_FILE = Path("data/processed/sentiment_pilot.csv")
OUTPUT_FILE = Path("data/processed/pilot_label_mismatch.csv")

# Membaca dataset
df = pd.read_csv(INPUT_FILE)

# Memastikan kolom tersedia
required_columns = ["comment", "sentiment_llm", "manual_sentiment"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Kolom berikut tidak ditemukan: {missing_columns}")

# Membersihkan format label
df["sentiment_llm"] = df["sentiment_llm"].astype(str).str.strip().str.lower()
df["manual_sentiment"] = df["manual_sentiment"].astype(str).str.strip().str.lower()

# Mengambil komentar yang labelnya berbeda
valid_labels = ["positif", "negatif", "netral"]

valid_df = df[
    df["sentiment_llm"].isin(valid_labels)
    & df["manual_sentiment"].isin(valid_labels)
].copy()

mismatch = valid_df[
    valid_df["sentiment_llm"] != valid_df["manual_sentiment"]
].copy()

# Menyimpan seluruh komentar yang berbeda
mismatch.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("=" * 60)
print("ANALISIS PERBEDAAN LABEL PILOT")
print("=" * 60)

print(f"\nJumlah data valid       : {len(valid_df)}")
print(f"Jumlah label berbeda    : {len(mismatch)}")
print(f"Persentase perbedaan    : {len(mismatch) / len(valid_df) * 100:.2f}%")

# Mengelompokkan berdasarkan pasangan label
print("\n--- Kelompok Perbedaan Label ---")

grouped = (
    mismatch.groupby(["manual_sentiment", "sentiment_llm"])
    .size()
    .reset_index(name="jumlah")
    .sort_values("jumlah", ascending=False)
)

print(grouped.to_string(index=False))

# Menampilkan contoh komentar dari setiap kelompok
print("\n--- Contoh Komentar yang Berbeda ---")

for _, row in grouped.iterrows():
    manual = row["manual_sentiment"]
    llm = row["sentiment_llm"]

    samples = mismatch[
        (mismatch["manual_sentiment"] == manual)
        & (mismatch["sentiment_llm"] == llm)
    ]

    print(f"\nManual: {manual.upper()} | Ollama: {llm.upper()}")
    print(f"Jumlah: {len(samples)}")

    for i, (_, sample) in enumerate(samples.head(5).iterrows(), start=1):
        print(f"\n{i}. {sample['comment']}")

print("\n" + "=" * 60)
print(f"Hasil lengkap disimpan di: {OUTPUT_FILE}")
print("=" * 60)