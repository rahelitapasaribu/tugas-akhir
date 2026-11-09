"""
Mengaudit kosakata dari dataset 10.000 komentar untuk menyusun kandidat dictionary normalisasi.
- word_frequency.csv: Seluruh kata unik beserta frekuensi dan contoh penggunaannya.
- normalization_candidates.csv: Kata yang perlu diperiksa, termasuk kata berulang dan kata yang muncul minimal 5 kali.
"""

import re
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

# Lokasi dataset
INPUT_PATH = Path("data/processed/comments_custom_10000.csv")
OUTPUT_DIR = Path("data/processed/vocabulary_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Membaca dataset
df = pd.read_csv(INPUT_PATH)

if "comment" not in df.columns:
    raise ValueError("Kolom 'comment' tidak ditemukan.")

df["comment"] = df["comment"].fillna("").astype(str)

# Menyimpan frekuensi kata dan contoh komentar
word_counts = Counter()
word_examples = defaultdict(list)

for comment in df["comment"]:
    # Mengubah teks menjadi huruf kecil untuk keperluan audit
    text = comment.lower()

    # Mengambil kata, termasuk kata berhuruf berulang
    words = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)

    for word in words:
        word_counts[word] += 1

        if len(word_examples[word]) < 3:
            if comment not in word_examples[word]:
                word_examples[word].append(comment)

# Membuat tabel frekuensi seluruh kata
frequency_data = []

for word, count in word_counts.most_common():
    frequency_data.append({
        "word": word,
        "frequency": count,
        "examples": " || ".join(word_examples[word])
    })

frequency_df = pd.DataFrame(frequency_data)

# Menandai kata yang berpotensi perlu diperiksa
def check_candidate(word):
    reasons = []

    if len(word) >= 15:
        reasons.append("kata_panjang")

    if re.search(r"(.)\1{2,}", word):
        reasons.append("huruf_berulang")

    if re.search(r"\d", word):
        reasons.append("mengandung_angka")

    return ", ".join(reasons)

frequency_df["review_reason"] = frequency_df["word"].apply(
    check_candidate
)

# Simpan seluruh frekuensi kata
frequency_df.to_csv(
    OUTPUT_DIR / "word_frequency.csv",
    index=False,
    encoding="utf-8-sig"
)

# Simpan kandidat kata yang perlu diperiksa
candidate_df = frequency_df[
    (frequency_df["review_reason"] != "") |
    (frequency_df["frequency"] >= 5)
].copy()

candidate_df.to_csv(
    OUTPUT_DIR / "normalization_candidates.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Audit kosakata selesai.")
print(f"Jumlah komentar : {len(df)}")
print(f"Jumlah kata unik: {len(frequency_df)}")
print(f"Kandidat review : {len(candidate_df)}")
print(f"Hasil tersimpan : {OUTPUT_DIR}")