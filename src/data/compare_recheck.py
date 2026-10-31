# Kegunaan: Membandingkan label manual, Ollama awal, dan hasil pemeriksaan ulang Qwen untuk menentukan komentar yang perlu ditinjau kembali.

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data/processed/pilot_recheck_suggestions.csv"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/pilot_comparison_report.csv"

df = pd.read_csv(INPUT_FILE)

columns = ["manual_sentiment", "sentiment_llm", "recheck_label"]

for col in columns:
    df[col] = df[col].astype(str).str.strip().str.lower()

# Membandingkan ketiga label
df["manual_vs_ollama"] = (
    df["manual_sentiment"] == df["sentiment_llm"]
)

df["manual_vs_recheck"] = (
    df["manual_sentiment"] == df["recheck_label"]
)

df["ollama_vs_recheck"] = (
    df["sentiment_llm"] == df["recheck_label"]
)

# Menentukan kelompok kesepakatan
def agreement(row):
    labels = [
        row["manual_sentiment"],
        row["sentiment_llm"],
        row["recheck_label"]
    ]

    if labels[0] == labels[1] == labels[2]:
        return "Ketiganya sepakat"

    if labels[0] == labels[1]:
        return "Manual dan Ollama sepakat"

    if labels[0] == labels[2]:
        return "Manual dan Qwen sepakat"

    if labels[1] == labels[2]:
        return "Ollama dan Qwen sepakat"

    return "Ketiganya berbeda"

df["agreement_group"] = df.apply(agreement, axis=1)

# Menampilkan ringkasan
print("=" * 55)
print("PERBANDINGAN TIGA LABEL SENTIMEN")
print("=" * 55)

print("\nKesepakatan antarlabel:")
for col in [
    "manual_vs_ollama",
    "manual_vs_recheck",
    "ollama_vs_recheck"
]:
    same = df[col].sum()
    total = len(df)

    print(
        f"{col}: {same}/{total} "
        f"({same / total * 100:.2f}%)"
    )

print("\nKelompok kesepakatan:")
print(df["agreement_group"].value_counts().to_string())

# Menyimpan laporan
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"\nLaporan disimpan di: {OUTPUT_FILE}")