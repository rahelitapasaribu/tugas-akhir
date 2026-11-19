# Melakukan preprocessing final pada 10.000 komentar YouTube.
# Membersihkan URL, tag HTML, karakter kontrol, dan spasi berlebih.
# Melakukan case folding dan normalisasi menggunakan 104 pasangan kata.
# Mempertahankan emoji, tanda baca, dan informasi sentimen.
# Menyimpan hasil tanpa menimpa dataset asli.

from pathlib import Path
from collections import Counter
import html
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "data" / "processed" / "comments_custom_10000.csv"
)

DICTIONARY_PATH = (
    ROOT / "data" / "processed" / "vocabulary_audit"
    / "kamusalay_extension"
    / "normalization_dictionary_final.csv"
)

OUTPUT_PATH = (
    ROOT / "data" / "processed" / "comments_preprocessed_final.csv"
)

REPORT_PATH = (
    ROOT / "results" / "metrics" / "preprocessing_final_report.csv"
)

def load_dictionary(path):
    # Membaca kamus gabungan dan memvalidasi isinya.
    df = pd.read_csv(
        path, encoding="utf-8-sig", dtype=str
    ).fillna("")

    df.columns = df.columns.str.strip().str.lower()

    required = {"word", "normalisasi"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Kolom kamus tidak sesuai: {df.columns.tolist()}"
        )

    dictionary = {}

    for _, row in df.iterrows():
        source = row["word"].strip().lower()
        target = row["normalisasi"].strip().lower()

        if not source or not target:
            raise ValueError("Terdapat pasangan kamus kosong.")

        if source in dictionary and dictionary[source] != target:
            raise ValueError(f"Konflik kamus: {source}")

        if source in dictionary:
            raise ValueError(f"Duplikat kamus: {source}")

        dictionary[source] = target

    return dictionary

def clean_comment(text):
    # Membersihkan teks sebelum normalisasi.
    if pd.isna(text):
        return ""

    text = str(text)

    # Mengubah HTML entities menjadi karakter aslinya.
    text = html.unescape(text)

    # Menghapus URL.
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Menghapus tag HTML tanpa menghapus teks di dalamnya.
    text = re.sub(
        r"</?[a-zA-Z][^>]*>",
        " ",
        text
    )

    # Menghapus karakter kontrol.
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)

    # Mengubah huruf menjadi lowercase.
    text = text.lower()

    # Merapikan spasi.
    text = re.sub(r"\s+", " ", text).strip()

    return text

def build_pattern(dictionary):
    # Mencocokkan kata utuh, bukan potongan kata.
    words = sorted(dictionary.keys(), key=len, reverse=True)

    alternatives = "|".join(
        re.escape(word) for word in words
    )

    return re.compile(
        rf"(?<!\w)(?:{alternatives})(?!\w)",
        flags=re.UNICODE
    )

def normalize_comment(text, dictionary, pattern, counts):
    # Mengganti kata sesuai kamus dan menghitung perubahan.
    def replace(match):
        source = match.group(0)
        counts[source] += 1
        return dictionary[source]

    return pattern.sub(replace, text)

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    if not DICTIONARY_PATH.exists():
        raise FileNotFoundError(DICTIONARY_PATH)

    df = pd.read_csv(
        INPUT_PATH, encoding="utf-8-sig"
    )

    if "comment" not in df.columns:
        raise ValueError("Kolom 'comment' tidak ditemukan.")

    dictionary = load_dictionary(DICTIONARY_PATH)
    pattern = build_pattern(dictionary)
    counts = Counter()

    # Menyimpan komentar asli untuk pembandingan.
    df["comment_original"] = df["comment"].fillna("")

    # Cleaning dan case folding.
    df["comment_cleaned"] = (
        df["comment_original"].apply(clean_comment)
    )

    # Normalisasi menggunakan kamus gabungan.
    df["comment_preprocessed"] = (
        df["comment_cleaned"].apply(
            lambda text: normalize_comment(
                text, dictionary, pattern, counts
            )
        )
    )

    # Menghitung perubahan yang terjadi.
    changed_normalization = (
        df["comment_cleaned"] != df["comment_preprocessed"]
    )

    changed_total = (
        df["comment_original"] != df["comment_preprocessed"]
    )

    empty_count = (
        df["comment_preprocessed"].str.strip().eq("").sum()
    )

    # Menyimpan dataset final.
    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # Menyimpan laporan penggunaan setiap pasangan kamus.
    report = pd.DataFrame([
        {
            "word": word,
            "normalisasi": target,
            "jumlah_penggantian": counts[word]
        }
        for word, target in dictionary.items()
    ])

    REPORT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    report.to_csv(
        REPORT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("Preprocessing final selesai.")
    print(f"Total komentar: {len(df)}")
    print(f"Pasangan kamus: {len(dictionary)}")
    print(
        "Pasangan kamus terpakai:",
        sum(count > 0 for count in counts.values())
    )
    print(
        "Komentar berubah karena normalisasi:",
        int(changed_normalization.sum())
    )
    print(
        "Komentar berubah secara keseluruhan:",
        int(changed_total.sum())
    )
    print(f"Total penggantian kata: {sum(counts.values())}")
    print(f"Komentar kosong: {empty_count}")
    print(f"Dataset: {OUTPUT_PATH}")
    print(f"Laporan: {REPORT_PATH}")

if __name__ == "__main__":
    main()