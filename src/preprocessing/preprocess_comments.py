"""Preprocessing komentar YouTube menggunakan kamus normalisasi v3.
Membersihkan teks, melakukan case folding, dan normalisasi kata tidak baku.
Mempertahankan emoji, tanda baca, negasi, dan ekspresi sentimen.
Menyimpan komentar asli dan hasil preprocessing untuk pemeriksaan. """

from pathlib import Path
import html
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data" / "processed" / "comments_custom_10000.csv"
DICTIONARY_PATH = (
    ROOT / "data" / "processed" / "vocabulary_audit"
    / "normalization_dictionary_v3.csv"
)
OUTPUT_PATH = ROOT / "data" / "processed" / "comments_preprocessed.csv"

TEXT_COLUMN = "comment"

# Membaca kamus normalisasi
def load_dictionary(path):
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    df.columns = df.columns.str.strip().str.lower()

    source_candidates = [
        "word", "kata", "kata_asli", "original_word",
        "nonstandard", "nonstandard_word", "original"
    ]

    target_candidates = [
        "normalization", "normalisasi", "kata_baku",
        "normalized_word", "replacement", "target"
    ]

    source_col = next(
        (col for col in source_candidates if col in df.columns),
        None
    )
    target_col = next(
        (col for col in target_candidates if col in df.columns),
        None
    )

    if source_col is None or target_col is None:
        raise ValueError(
            "Kolom kamus tidak dikenali. "
            f"Kolom yang tersedia: {df.columns.tolist()}"
        )

    dictionary = {}

    for _, row in df.iterrows():
        source = row[source_col].strip().lower()
        target = row[target_col].strip().lower()

        if not source or not target or source == target:
            continue

        if source in dictionary and dictionary[source] != target:
            raise ValueError(
                f"Konflik normalisasi: {source}"
            )

        dictionary[source] = target

    return dictionary


# Membersihkan karakter yang tidak diperlukan
def clean_text(text):
    if pd.isna(text):
        return ""

    text = html.unescape(str(text))

    # Menghapus URL
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Menghapus karakter kontrol tanpa menghapus emoji
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)

    # Case folding
    text = text.lower()

    # Merapikan spasi
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Membuat pola pencocokan kata utuh
def build_normalization_pattern(dictionary):
    if not dictionary:
        return None

    words = sorted(dictionary, key=len, reverse=True)
    alternatives = "|".join(re.escape(word) for word in words)

    return re.compile(
        rf"(?<!\w)(?:{alternatives})(?!\w)",
        flags=re.UNICODE
    )


# Melakukan normalisasi berdasarkan kamus
def normalize_text(text, dictionary, pattern):
    if not text or pattern is None:
        return text

    return pattern.sub(
        lambda match: dictionary[match.group(0)],
        text
    )


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {INPUT_PATH}"
        )

    if not DICTIONARY_PATH.exists():
        raise FileNotFoundError(
            f"Kamus tidak ditemukan: {DICTIONARY_PATH}"
        )

    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

    if TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"Kolom '{TEXT_COLUMN}' tidak ditemukan. "
            f"Kolom tersedia: {df.columns.tolist()}"
        )

    dictionary = load_dictionary(DICTIONARY_PATH)
    pattern = build_normalization_pattern(dictionary)

    # Menyimpan teks asli tanpa mengubah dataset sumber
    df["comment_original"] = df[TEXT_COLUMN].fillna("")

    # Tahap pembersihan dan case folding
    df["comment_cleaned"] = df["comment_original"].apply(clean_text)

    # Tahap normalisasi
    df["comment_preprocessed"] = df["comment_cleaned"].apply(
        lambda text: normalize_text(text, dictionary, pattern)
    )

    # Memeriksa perubahan akibat normalisasi saja
    changed = (
        df["comment_cleaned"] != df["comment_preprocessed"]
    )

    empty = df["comment_preprocessed"].str.strip().eq("")

    # Menyimpan seluruh kolom dataset dan hasil preprocessing
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("Preprocessing selesai.")
    print(f"Total komentar: {len(df)}")
    print(f"Pasangan kamus aktif: {len(dictionary)}")
    print(f"Komentar berubah karena normalisasi: {changed.sum()}")
    print(f"Komentar kosong setelah preprocessing: {empty.sum()}")
    print(f"Hasil disimpan di: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()