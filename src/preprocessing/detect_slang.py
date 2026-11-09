
# Mendeteksi dan mengklasifikasikan seluruh kosakata menggunakan Qwen melalui Ollama.

import ast
import json
from pathlib import Path

import pandas as pd
import ollama


# ============================================================
# 1. KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR / "data" / "processed"
    / "vocabulary_audit" / "word_frequency.csv"
)

OUTPUT_DIR = (
    BASE_DIR / "data" / "processed"
    / "vocabulary_audit"
)

OUTPUT_FILE = OUTPUT_DIR / "llm_dictionary_candidates.csv"

MODEL_NAME = "qwen2.5:3b"
BATCH_SIZE = 15
MAX_EXAMPLE_LENGTH = 180


# ============================================================
# 2. KAMUS ACUAN NORMALISASI
# ============================================================

# Kata yang memiliki bentuk normal jelas.
# Kata-kata ini akan diperbaiki secara otomatis pada hasil klasifikasi,
# bukan langsung diterapkan ke dataset komentar.

NORMALIZATION_OVERRIDES = {
    "gk": "tidak",
    "gak": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "enggak": "tidak",
    "blm": "belum",
    "udh": "sudah",
    "udah": "sudah",
    "yg": "yang",
    "dgn": "dengan",
    "krn": "karena",
    "bgt": "banget",
    "skrg": "sekarang",
    "org": "orang",
    "gmn": "bagaimana",
    "kyk": "seperti",
    "klo": "kalau",
    "kl": "kalau",
    "dr": "dari",
    "sm": "sama",
    "tp": "tapi",
    "jd": "jadi",
    "tdk": "tidak",
    "sy": "saya",
    "ak": "aku",
    "aja": "saja",
    "trs": "terus",
    "krna": "karena"
}


# ============================================================
# 3. MEMBACA DAN MENYESUAIKAN DATA
# ============================================================

def find_column(dataframe, possible_names):
    """Mencari nama kolom berdasarkan beberapa kemungkinan nama."""

    for name in possible_names:
        if name in dataframe.columns:
            return name

    return None


def parse_examples(value):
    """Mengubah contoh komentar menjadi daftar teks."""

    if value is None:
        return []

    if isinstance(value, list):
        examples = value

    elif pd.isna(value):
        return []

    else:
        value = str(value).strip()

        try:
            examples = ast.literal_eval(value)

            if not isinstance(examples, list):
                examples = [value]

        except (ValueError, SyntaxError):
            examples = [value]

    cleaned_examples = []

    for example in examples:
        example = str(example).strip()

        if example:
            cleaned_examples.append(
                example[:MAX_EXAMPLE_LENGTH]
            )

    return cleaned_examples[:3]


def load_vocabulary():
    """Membaca seluruh kosakata unik dari hasil audit vocabulary."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File input tidak ditemukan: {INPUT_FILE}"
        )

    dataframe = pd.read_csv(INPUT_FILE)

    word_column = find_column(
        dataframe, ["word", "token", "kata"]
    )

    frequency_column = find_column(
        dataframe, ["frequency", "count", "freq", "jumlah"]
    )

    examples_column = find_column(
        dataframe,
        ["examples", "example_comments",
         "contoh_komentar", "examples_comments"]
    )

    if word_column is None:
        raise ValueError(
            f"Kolom kata tidak ditemukan: {list(dataframe.columns)}"
        )

    if frequency_column is None:
        raise ValueError("Kolom frekuensi tidak ditemukan.")

    if examples_column is None:
        raise ValueError("Kolom contoh komentar tidak ditemukan.")

    dataframe = dataframe.rename(
        columns={
            word_column: "word",
            frequency_column: "frequency",
            examples_column: "examples"
        }
    )

    dataframe = dataframe[
        ["word", "frequency", "examples"]
    ].copy()

    dataframe["word"] = (
        dataframe["word"].astype(str).str.strip()
    )

    dataframe = dataframe[dataframe["word"] != ""]

    dataframe["frequency"] = pd.to_numeric(
        dataframe["frequency"],
        errors="coerce"
    ).fillna(0).astype(int)

    dataframe["examples"] = (
        dataframe["examples"].apply(parse_examples)
    )

    dataframe = dataframe.drop_duplicates(
        subset=["word"]
    ).reset_index(drop=True)

    return dataframe


# ============================================================
# 4. INSTRUKSI KLASIFIKASI QWEN
# ============================================================

SYSTEM_PROMPT = """
Kamu adalah asisten linguistik yang membantu mengidentifikasi
kosakata dalam komentar media sosial berbahasa Indonesia.

Tugasmu adalah mengklasifikasikan setiap kata berdasarkan
frekuensi dan contoh penggunaannya.

ATURAN PENTING:

1. Jangan menganggap semua kata tidak baku sebagai typo.
2. Kenali nama orang, tokoh, tempat, akun, organisasi,
   dan istilah khusus.
3. Nama seperti soohyun, saeron, kimsoohyun,
   dan variasi nama tokoh harus dipertahankan.
4. Jangan mengubah nama orang menjadi kata bahasa Indonesia.
5. Pertahankan kata asing dan istilah fandom yang bermakna.
6. Perhatikan konteks contoh komentar.
7. Jangan menghilangkan atau mengubah negasi.
8. Pertahankan ekspresi emosi dan intensitas jika perubahan
   dapat mengubah makna sentimen.
9. Jika makna tidak jelas, pilih review_manual.
10. Jangan menebak bentuk normalisasi.

CONTOH NORMALISASI:

gk -> tidak
gak -> tidak
ga -> tidak
nggak -> tidak
ngga -> tidak
blm -> belum
udh -> sudah
yg -> yang
dgn -> dengan
krn -> karena
bgt -> banget
skrg -> sekarang
org -> orang
gmn -> bagaimana
kyk -> seperti
klo -> kalau

Gunakan contoh tersebut sebagai acuan.
Jika menemukan variasi lain yang jelas memiliki makna sama,
berikan bentuk normalisasinya.

KATEGORI:

- singkatan
- bahasa_gaul
- variasi_fonetik
- typo
- repetisi_karakter
- nama_entitas
- kata_baku
- istilah_khusus
- tidak_diketahui

KEPUTUSAN:

- normalisasi
- pertahankan
- review_manual

ATURAN KEPUTUSAN:

normalisasi:
Kata tidak baku atau salah tulis yang bentuk normalnya
jelas dan maknanya tidak berubah.

pertahankan:
Kata baku, nama orang, nama entitas, istilah khusus,
atau kata yang tidak perlu diubah.

review_manual:
Konteks atau maknanya belum cukup jelas.

Jika keputusan normalisasi, isi suggested_normalization
dengan bentuk normal yang disarankan.

Jika keputusan pertahankan atau review_manual,
isi suggested_normalization dengan string kosong.

Confidence hanya berupa:
tinggi, sedang, atau rendah.

Keluarkan JSON valid dengan struktur:

{
  "results": [
    {
      "word": "kata",
      "category": "kategori",
      "suggested_normalization": "",
      "decision": "pertahankan",
      "confidence": "tinggi",
      "reason": "alasan singkat"
    }
  ]
}

Berikan hasil untuk setiap kata.
Jangan menambahkan atau menghilangkan kata.
"""


# ============================================================
# 5. MEMPROSES SATU BATCH KOSAKATA
# ============================================================

def classify_batch(batch):
    """Meminta Qwen mengklasifikasikan satu kelompok kosakata."""

    vocabulary_data = []

    for _, row in batch.iterrows():
        vocabulary_data.append({
            "word": row["word"],
            "frequency": int(row["frequency"]),
            "examples": row["examples"]
        })

    user_prompt = f"""
Klasifikasikan seluruh kosakata berikut.

Data kosakata:

{json.dumps(vocabulary_data, ensure_ascii=False, indent=2)}

Perhatikan:
- Periksa contoh komentar.
- Nama orang dan entitas khusus harus dipertahankan.
- Berikan bentuk normalisasi jika jelas.
- Jangan memaksakan normalisasi pada kata ambigu.
- Berikan hasil untuk setiap kata.
- Keluarkan JSON valid saja.
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        format="json",
        options={
            "temperature": 0
        }
    )

    content = response["message"]["content"]

    result_json = json.loads(content)

    return result_json.get("results", [])


# ============================================================
# 6. VALIDASI DAN PERBAIKAN HASIL QWEN
# ============================================================

VALID_CATEGORIES = {
    "singkatan",
    "bahasa_gaul",
    "variasi_fonetik",
    "typo",
    "repetisi_karakter",
    "nama_entitas",
    "kata_baku",
    "istilah_khusus",
    "tidak_diketahui"
}

VALID_DECISIONS = {
    "normalisasi",
    "pertahankan",
    "review_manual"
}


def validate_results(batch, llm_results):
    """Memvalidasi hasil Qwen dan menerapkan kamus acuan."""

    expected_words = set(batch["word"].tolist())

    result_by_word = {}

    for item in llm_results:

        word = str(item.get("word", "")).strip()

        if word not in expected_words:
            continue

        if word in result_by_word:
            continue

        category = str(
            item.get("category", "tidak_diketahui")
        ).strip().lower()

        decision = str(
            item.get("decision", "review_manual")
        ).strip().lower()

        normalization = str(
            item.get("suggested_normalization", "")
        ).strip()

        confidence = str(
            item.get("confidence", "rendah")
        ).strip().lower()

        reason = str(
            item.get("reason", "")
        ).strip()

        if category not in VALID_CATEGORIES:
            category = "tidak_diketahui"
            decision = "review_manual"

        if decision not in VALID_DECISIONS:
            decision = "review_manual"

        # ----------------------------------------------------
        # PERBAIKAN: KAMUS ACUAN NORMALISASI
        # ----------------------------------------------------

        word_key = word.lower().strip()

        if word_key in NORMALIZATION_OVERRIDES:

            category = "singkatan"

            normalization = (
                NORMALIZATION_OVERRIDES[word_key]
            )

            decision = "normalisasi"
            confidence = "tinggi"

            reason = (
                "Bentuk slang memiliki padanan normal "
                "yang jelas berdasarkan kamus acuan."
            )

        elif decision != "normalisasi":
            normalization = ""

        if decision == "normalisasi" and not normalization:
            decision = "review_manual"

        result_by_word[word] = {
            "category": category,
            "suggested_normalization": normalization,
            "decision": decision,
            "confidence": confidence,
            "reason": reason
        }

    validated = []

    for _, row in batch.iterrows():

        word = row["word"]

        result = result_by_word.get(word)

        if result is None:
            result = {
                "category": "tidak_diketahui",
                "suggested_normalization": "",
                "decision": "review_manual",
                "confidence": "rendah",
                "reason": (
                    "Hasil klasifikasi tidak diberikan oleh Qwen."
                )
            }

        validated.append({
            "word": word,
            "frequency": row["frequency"],
            "examples": json.dumps(
                row["examples"],
                ensure_ascii=False
            ),
            **result
        })

    return validated


# ============================================================
# 7. PROSES SELURUH KOSAKATA
# ============================================================

def main():
    """Menjalankan klasifikasi seluruh kosakata dan menyimpan hasil."""

    print("=" * 60)
    print("DETEKSI KOSAKATA DENGAN QWEN")
    print("=" * 60)

    vocabulary = load_vocabulary()

    total_words = len(vocabulary)

    print(f"Total kosakata unik : {total_words}")
    print(f"Model               : {MODEL_NAME}")
    print(f"Ukuran batch        : {BATCH_SIZE}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_results = []

    total_batches = (
        total_words + BATCH_SIZE - 1
    ) // BATCH_SIZE

    for start in range(0, total_words, BATCH_SIZE):

        batch = vocabulary.iloc[
            start:start + BATCH_SIZE
        ].copy()

        batch_number = (start // BATCH_SIZE) + 1

        print(
            f"\nBatch {batch_number}/{total_batches} "
            f"({len(batch)} kata)"
        )

        try:
            llm_results = classify_batch(batch)

            validated = validate_results(
                batch,
                llm_results
            )

            all_results.extend(validated)

            print(
                f"Berhasil diproses: {len(validated)} kata"
            )

        except Exception as error:

            print(f"Batch gagal: {error}")

            failed_results = validate_results(
                batch,
                []
            )

            all_results.extend(failed_results)

        # Simpan hasil setiap batch.
        temporary_df = pd.DataFrame(all_results)

        temporary_df.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig"
        )

    # ========================================================
    # 8. RINGKASAN HASIL
    # ========================================================

    final_df = pd.DataFrame(all_results)

    final_df = final_df.drop_duplicates(
        subset=["word"],
        keep="last"
    )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("KLASIFIKASI SELESAI")
    print("=" * 60)

    print(f"Total kosakata : {len(final_df)}")

    print(
        "Normalisasi    :",
        (final_df["decision"] == "normalisasi").sum()
    )

    print(
        "Pertahankan    :",
        (final_df["decision"] == "pertahankan").sum()
    )

    print(
        "Review manual  :",
        (final_df["decision"] == "review_manual").sum()
    )

    print(f"\nHasil disimpan di:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()