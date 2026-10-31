# Kegunaan: Membuat daftar prioritas pemeriksaan manual berdasarkan perbandingan label manual, Ollama, dan Qwen.

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data/processed/pilot_comparison_report.csv"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/pilot_review_queue.csv"

df = pd.read_csv(INPUT_FILE)

# Mengambil komentar yang rekomendasi Qwen berbeda dari label manual
review = df[
    df["manual_sentiment"] != df["recheck_label"]
].copy()

# Menentukan prioritas pemeriksaan
def determine_priority(row):
    if row["agreement_group"] == "Ketiganya berbeda":
        return "Prioritas 1 - Ketiganya berbeda"

    if row["agreement_group"] == "Ollama dan Qwen sepakat":
        return "Prioritas 2 - Kedua model sepakat"

    return "Prioritas 3 - Perlu pemeriksaan"

review["review_priority"] = review.apply(
    determine_priority,
    axis=1
)

# Urutan prioritas
priority_order = {
    "Prioritas 1 - Ketiganya berbeda": 1,
    "Prioritas 2 - Kedua model sepakat": 2,
    "Prioritas 3 - Perlu pemeriksaan": 3
}

review["_order"] = review["review_priority"].map(priority_order)

review = review.sort_values("_order").drop(columns="_order")

# Menambahkan kolom keputusan manual
review["final_sentiment"] = ""
review["review_notes"] = ""

# Menyimpan hasil
review.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("DAFTAR PRIORITAS PEMERIKSAAN MANUAL")
print("=" * 50)
print(f"Total komentar perlu diperiksa: {len(review)}")
print("\nJumlah berdasarkan prioritas:")
print(review["review_priority"].value_counts().to_string())

print(f"\nFile disimpan di: {OUTPUT_FILE}")