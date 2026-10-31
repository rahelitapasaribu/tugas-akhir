# Memeriksa ulang komentar yang labelnya berbeda menggunakan pedoman anotasi dari file Markdown.

import pandas as pd
import ollama
import json
import time
from pathlib import Path

# Konfigurasi lokasi file berdasarkan folder utama proyek
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data/processed/pilot_label_mismatch.csv"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/pilot_recheck_suggestions.csv"
GUIDELINE_FILE = PROJECT_ROOT / "docs/annotation_guideline.md"

MODEL_NAME = "qwen2.5:3b"
VALID_LABELS = ["positif", "negatif", "netral"]

# Membaca pedoman anotasi dari file Markdown
if not GUIDELINE_FILE.exists():
    raise FileNotFoundError(
        f"File pedoman anotasi tidak ditemukan: {GUIDELINE_FILE}"
    )

GUIDELINE = GUIDELINE_FILE.read_text(encoding="utf-8")

# Memeriksa file input
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"File tidak ditemukan: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "comment",
    "manual_sentiment",
    "sentiment_llm"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Kolom tidak ditemukan: {column}")

# Memuat hasil sebelumnya agar proses bisa dilanjutkan
# Memuat hasil sebelumnya agar proses bisa dilanjutkan
if OUTPUT_FILE.exists():
    previous = pd.read_csv(OUTPUT_FILE)

    if "recheck_label" not in previous.columns:
        previous["recheck_label"] = None

    if "recheck_reason" not in previous.columns:
        previous["recheck_reason"] = None

    df = df.merge(
        previous[["comment", "recheck_label", "recheck_reason"]].drop_duplicates(
            subset=["comment"]
        ),
        on="comment",
        how="left"
    )
else:
    df["recheck_label"] = None
    df["recheck_reason"] = None

# Mengubah tipe kolom agar dapat menyimpan teks
df["recheck_label"] = df["recheck_label"].astype("object")
df["recheck_reason"] = df["recheck_reason"].astype("object")

# Proses pemeriksaan ulang
for index, row in df.iterrows():

    # Lewati komentar yang sudah diperiksa
    if pd.notna(row["recheck_label"]) and str(row["recheck_label"]).lower() in VALID_LABELS:
        continue

    comment = str(row["comment"])
    manual_label = str(row["manual_sentiment"]).lower()
    ollama_label = str(row["sentiment_llm"]).lower()

    prompt = f"""
{GUIDELINE}

Tugas:
Tentukan sentimen komentar secara independen berdasarkan
isi dan konteksnya.

Jangan otomatis mengikuti label manual atau label Ollama.
Gunakan kedua label tersebut hanya sebagai informasi pembanding.

Komentar:
{comment}

Label manual sebelumnya: {manual_label}
Label Ollama sebelumnya: {ollama_label}

Berikan jawaban dalam JSON dengan format:
{{
    "label": "positif/negatif/netral",
    "alasan": "Alasan singkat berdasarkan konteks komentar"
}}
"""

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            format="json",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={"temperature": 0}
        )

        result = json.loads(response["message"]["content"])

        suggested_label = str(result.get("label", "")).strip().lower()
        reason = str(result.get("alasan", "")).strip()

        if suggested_label not in VALID_LABELS:
            raise ValueError("Label hasil Ollama tidak valid")

        df.at[index, "recheck_label"] = suggested_label
        df.at[index, "recheck_reason"] = reason

        print(f"[{index + 1}/{len(df)}] Selesai: {suggested_label}")

    except Exception as error:
        print(f"[{index + 1}/{len(df)}] Gagal: {error}")

    # Simpan progres setiap komentar
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    time.sleep(0.5)

print("\nPemeriksaan ulang selesai.")
print(f"Hasil disimpan di: {OUTPUT_FILE}")

print("\nDistribusi rekomendasi label:")
print(df["recheck_label"].value_counts(dropna=False))

print("\nJumlah rekomendasi berbeda dari label manual:")
print(
    (
        df["recheck_label"].astype(str).str.lower()
        != df["manual_sentiment"].astype(str).str.lower()
    ).sum()
)