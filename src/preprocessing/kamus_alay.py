# Mencari kandidat normalisasi tambahan dari kamus alay eksternal.
# Menggunakan kata yang benar-benar muncul dalam dataset komentar.
# Tidak mengubah kamus v3 dan tidak melakukan preprocessing.
# Mengecualikan pemanjangan huruf dan kata ambigu yang sudah diketahui.

from pathlib import Path
from collections import Counter
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/comments_custom_10000.csv"
V3_PATH = ROOT / "data/processed/vocabulary_audit/normalization_dictionary_v3.csv"
ALAY_PATH = ROOT / "data/processed/vocabulary_audit/kamusalay.csv"

OUTPUT_DIR = ROOT / "data/processed/vocabulary_audit/kamusalay_extension"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Kata yang sebelumnya bermasalah atau ambigu.
BLOCKED = {
    "dah", "kek", "team", "kalaupun",
    "ajaaa", "yaaaa", "akhirnyaaa", "akhirnyaa",
    "iniii", "jugaa", "bgttt",
    "sm", "cm", "km", "dr", "no", "ad",
    "yg",  # jika belum ada di v3
}

# Fungsi untuk mencari kolom dalam file kamus v3.
def find_column(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Kolom tidak ditemukan. Tersedia: {list(columns)}")

def main():
    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    if "comment" not in data.columns:
        raise ValueError("Kolom comment tidak ditemukan.")

    v3 = pd.read_csv(V3_PATH, encoding="utf-8-sig", dtype=str)
    v3.columns = v3.columns.str.strip().str.lower()

    source_col = find_column(
        v3.columns,
        ["word", "kata", "kata_asli", "original_word",
         "nonstandard", "nonstandard_word", "original"]
    )
    target_col = find_column(
        v3.columns,
        ["normalization", "normalisasi", "kata_baku",
         "normalized_word", "replacement", "target"]
    )

    # Mengumpulkan seluruh kata dalam dataset.
    comments = data["comment"].fillna("").astype(str).tolist()
    frequency = Counter()

    for comment in comments:
        comment = re.sub(
            r"https?://\S+|www\.\S+", " ", comment.lower()
        )
        words = re.findall(r"\b\w+\b", comment, flags=re.UNICODE)
        frequency.update(words)

    existing_words = set(
        v3[source_col].dropna().str.strip().str.lower()
    )

    # Membaca kamus alay eksternal.
    alay = pd.read_csv(ALAY_PATH, encoding="utf-8-sig", dtype=str)
    alay.columns = alay.columns.str.strip().str.lower()

    if not {"slang", "formal"}.issubset(alay.columns):
        raise ValueError("Kamus alay harus memiliki kolom slang dan formal.")

    alay["slang"] = alay["slang"].fillna("").str.strip().str.lower()
    alay["formal"] = alay["formal"].fillna("").str.strip().str.lower()

    # Menghindari usulan yang saling bertentangan.
    conflicts = alay.groupby("slang")["formal"].nunique()
    conflicting_words = set(conflicts[conflicts > 1].index)

    results = []

    for _, row in alay.iterrows():
        slang = row["slang"]
        formal = row["formal"]

        if not slang or not formal or slang == formal:
            continue

        if slang in existing_words:
            continue

        if slang in BLOCKED or slang in conflicting_words:
            continue

        # Kandidat kata tunggal yang dapat dicocokkan dengan aman.
        if not re.fullmatch(r"[a-z0-9]+", slang):
            continue

        if len(slang) < 2:
            continue

        freq = frequency.get(slang, 0)
        if freq < 3:
            continue

        # Tidak langsung menerima bentuk pemanjangan huruf.
        categories = " ".join(
            str(row.get(col, ""))
            for col in ["category1", "category2", "category3"]
        ).lower()

        if "elongasi" in categories:
            continue

        # Mengambil satu contoh dari komentar asli.
        pattern = re.compile(
            rf"(?<!\w){re.escape(slang)}(?!\w)",
            flags=re.IGNORECASE
        )

        example = next(
            (comment for comment in comments if pattern.search(comment)),
            ""
        )

        results.append({
            "slang": slang,
            "formal": formal,
            "frequency": freq,
            "contoh_komentar": example,
            "sumber": "Salsabila et al. (2018)",
            "status": "kandidat_belum_divalidasi"
        })

    result = pd.DataFrame(
        results,
        columns=[
            "slang", "formal", "frequency",
            "contoh_komentar", "sumber", "status"
        ]
    )

    result = (
        result.drop_duplicates(subset=["slang", "formal"])
        .sort_values("frequency", ascending=False)
        .reset_index(drop=True)
    )

    # Menyimpan seluruh kandidat dan 50 kandidat prioritas.
    all_path = OUTPUT_DIR / "alay_candidates_all.csv"
    priority_path = OUTPUT_DIR / "alay_candidates_priority_50.csv"

    result.to_csv(all_path, index=False, encoding="utf-8-sig")
    result.head(50).to_csv(
        priority_path, index=False, encoding="utf-8-sig"
    )

    print("Seleksi kandidat kamus alay selesai.")
    print(f"Jumlah komentar: {len(comments)}")
    print(f"Kamus utama v3: {len(existing_words)} kata")
    print(f"Kandidat tambahan: {len(result)}")
    print(f"Kandidat prioritas: {min(50, len(result))}")
    print(f"Hasil disimpan di: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()