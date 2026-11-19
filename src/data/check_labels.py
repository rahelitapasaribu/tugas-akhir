# Memeriksa kualitas label sentimen awal yang dihasilkan Qwen.
# Menghitung distribusi dan kelengkapan label pada 10.000 komentar.
# Mendeteksi komentar identik dengan label yang berbeda.
# Menyiapkan sampel anotasi manual dari masing-masing kelas.
# Tidak mengubah label atau dataset asli.

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data/processed/comments_preprocessed_final.csv"
OUTPUT_DIR = ROOT / "data/processed/label_audit"

VALID_LABELS = {
    "positif": "Positif",
    "negatif": "Negatif",
    "netral": "Netral"
}

SAMPLE_PER_CLASS = 50
RANDOM_STATE = 42


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        INPUT_PATH,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False
    )

    required = [
        "comment_id",
        "comment_original",
        "comment_preprocessed",
        "sentiment_qwen"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}")

    # Menyeragamkan bentuk penulisan label untuk pemeriksaan.
    raw_labels = df["sentiment_qwen"].str.strip()

    df["label_awal"] = (
        raw_labels.str.lower().map(VALID_LABELS).fillna("")
    )

    invalid_mask = (
        raw_labels.ne("") & df["label_awal"].eq("")
    )

    # Menghitung distribusi label.
    distribution = (
        df["label_awal"]
        .replace("", "Kosong/Tidak dikenal")
        .value_counts()
        .rename_axis("label")
        .reset_index(name="jumlah")
    )

    distribution["persentase"] = (
        distribution["jumlah"] / len(df) * 100
    ).round(2)

    distribution.to_csv(
        OUTPUT_DIR / "label_distribution.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Menyimpan label tidak dikenal untuk diperiksa.
    df.loc[invalid_mask].to_csv(
        OUTPUT_DIR / "invalid_labels.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Mendeteksi teks identik yang memiliki label berbeda.
    nonempty = df[
        df["comment_preprocessed"].str.strip().ne("")
        & df["label_awal"].ne("")
    ].copy()

    label_counts = (
        nonempty.groupby("comment_preprocessed")["label_awal"]
        .nunique()
    )

    conflicting_texts = set(
        label_counts[label_counts > 1].index
    )

    conflicts = nonempty[
        nonempty["comment_preprocessed"].isin(
            conflicting_texts
        )
    ].copy()

    conflicts.to_csv(
        OUTPUT_DIR / "label_conflicts.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Mengambil sampel acak seimbang tanpa mengubah label asli.
    samples = []

    for label in VALID_LABELS.values():
        subset = df[df["label_awal"] == label]

        sample = subset.sample(
            n=min(SAMPLE_PER_CLASS, len(subset)),
            random_state=RANDOM_STATE
        ).copy()

        samples.append(sample)

    review = pd.concat(samples, ignore_index=True)

    # Membuat lembar validasi manual.
    columns = [
        "comment_id",
        "comment_original",
        "comment_preprocessed",
        "sentiment_qwen",
    ]

    if "reason_qwen" in review.columns:
        columns.append("reason_qwen")

    review = review[columns].copy()
    review["label_manual"] = ""
    review["status_validasi"] = ""
    review["catatan"] = ""

    review.to_csv(
        OUTPUT_DIR / "label_review_150.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("Pemeriksaan label awal selesai.")
    print(f"Total komentar: {len(df)}")
    print("\nDistribusi label:")
    print(distribution.to_string(index=False))
    print(f"\nLabel kosong: {raw_labels.eq('').sum()}")
    print(f"Label tidak dikenal: {invalid_mask.sum()}")
    print(f"Teks dengan konflik label: {len(conflicting_texts)}")
    print(f"Sampel review: {len(review)}")
    print(f"Hasil disimpan di: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()