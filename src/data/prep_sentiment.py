"""
Tujuan: Menyiapkan sampel pilot komentar berbahasa Indonesia untuk anotasi sentimen oleh LLM.
Input: data/processed/comments_lang_candidates.csv
Output: data/processed/sentiment_pilot.csv
Catatan: Sampel diambil hanya dari language_status == "candidate_id"; komentar kosong dan ID duplikat dibuang.
"""

from pathlib import Path
import pandas as pd

# 1. Konfigurasi
INPUT_PATH = Path("data/processed/comments_lang_candidates.csv")
OUTPUT_PATH = Path("data/processed/sentiment_pilot.csv")
SAMPLE_SIZE = 300
RANDOM_STATE = 42


def main():
    # 2. Pastikan file input tersedia
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"File input tidak ditemukan: {INPUT_PATH.resolve()}\n"
            "Jalankan proses finalisasi bahasa terlebih dahulu."
        )

    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

    # 3. Validasi kolom yang dibutuhkan
    required_columns = {"comment_id", "comment", "language_status"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {sorted(missing_columns)}. "
            f"Kolom tersedia: {list(df.columns)}"
        )

    # 4. Pilih kandidat bahasa Indonesia yang belum perlu ditinjau ulang
    df["comment"] = df["comment"].fillna("").astype(str).str.strip()
    df["language_status"] = (
        df["language_status"].fillna("").astype(str).str.strip()
    )

    candidates = df[
        (df["language_status"] == "candidate_id")
        & (df["comment"] != "")
        & (df["comment_id"].notna())
    ].copy()

    # Satu comment_id hanya boleh muncul sekali dalam sampel
    candidates = candidates.drop_duplicates(subset=["comment_id"])

    if candidates.empty:
        raise ValueError(
            "Tidak ada komentar dengan language_status == 'candidate_id'. "
            "Periksa isi kolom language_status pada file input."
        )

    # Jika data tersedia kurang dari 300, gunakan seluruh data yang tersedia
    sample_size = min(SAMPLE_SIZE, len(candidates))
    pilot = candidates.sample(n=sample_size, random_state=RANDOM_STATE)

    # 5. Siapkan kolom anotasi sentimen
    # Kolom sentiment_llm diisi oleh hasil anotasi LLM pada tahap berikutnya.
    # Kolom manual_sentiment disediakan untuk label pemeriksaan manual.
    pilot = pilot[
        ["comment_id", "comment", "language_status"]
    ].copy()
    pilot["sentiment_llm"] = ""
    pilot["manual_sentiment"] = ""
    pilot["review_notes"] = ""

    # 6. Simpan file pilot
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pilot.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("=== SENTIMENT PILOT READY ===")
    print(f"Jumlah kandidat candidate_id: {len(candidates):,}")
    print(f"Jumlah komentar pilot: {len(pilot):,}")
    print(f"File tersimpan di: {OUTPUT_PATH.resolve()}")
    print("\nKolom anotasi:")
    print("- sentiment_llm: label hasil LLM (positif/negatif/netral)")
    print("- manual_sentiment: label hasil pemeriksaan manual")
    print("- review_notes: alasan/catatan jika label meragukan")


if __name__ == "__main__":
    main()