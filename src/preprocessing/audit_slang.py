
# Audit hasil klasifikasi kosakata Qwen dan tandai hasil yang perlu diperiksa manual.

import json
from pathlib import Path
import pandas as pd

# KONFIGURASI
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vocabulary_audit"
    / "llm_dictionary_candidates.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vocabulary_audit"
    / "audit_results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Kata umum bahasa Indonesia yang seharusnya tidak perlu
# dinormalisasi dalam konteks penggunaan umum.
# Daftar ini hanya untuk MENANDAI, bukan mengubah keputusan Qwen.

COMMON_WORDS = {
    "dan", "di", "ini", "itu", "dia", "ada", "dari",
    "yang", "untuk", "dengan", "ke", "pada", "akan",
    "sudah", "belum", "juga", "atau", "karena", "kalau",
    "jika", "dalam", "oleh", "sebagai", "saya", "aku",
    "kamu", "kami", "kita", "mereka", "bisa", "tidak",
    "lebih", "sangat", "hanya", "masih", "saat", "sampai",
    "jadi", "ada", "apa", "siapa", "mana", "bagaimana",
    "mengapa", "ketika", "setelah", "sebelum", "tanpa",
    "pun", "nya", "lah", "kah", "kan", "mas", "mbak",
    "pak", "bu"
}

VALID_DECISIONS = {
    "normalisasi",
    "pertahankan",
    "review_manual"
}

# FUNGSI
def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()

def parse_examples(value):
    """
    Mengubah kolom examples berbentuk JSON menjadi teks
    agar lebih mudah dibaca dalam hasil audit.
    """
    if pd.isna(value):
        return ""

    try:
        parsed = json.loads(value)

        if isinstance(parsed, list):
            return " || ".join(str(x) for x in parsed)

    except (json.JSONDecodeError, TypeError):
        pass

    return str(value)

def audit_row(row):
    flags = []

    word = clean_text(row["word"]).lower()
    decision = clean_text(row["decision"]).lower()
    category = clean_text(row["category"]).lower()
    suggestion = clean_text(row["suggested_normalization"])
    reason = clean_text(row["reason"]).lower()

    frequency = int(row["frequency"])

    # 1. Keputusan tidak valid
    if decision not in VALID_DECISIONS:
        flags.append("keputusan_tidak_valid")

    # 2. Normalisasi tanpa saran
    if decision == "normalisasi" and not suggestion:
        flags.append("normalisasi_tanpa_saran")

    # 3. Saran normalisasi pada keputusan pertahankan
    if decision == "pertahankan" and suggestion:
        flags.append("pertahankan_memiliki_saran")

    # 4. Kata umum masuk review manual
    if word in COMMON_WORDS and decision == "review_manual":
        flags.append("kata_umum_review_manual")

    # 5. Kata umum justru dinormalisasi
    if word in COMMON_WORDS and decision == "normalisasi":
        flags.append("kata_umum_dinormalisasi")

    # 6. Review manual dengan frekuensi tinggi
    if decision == "review_manual" and frequency >= 100:
        flags.append("review_manual_frekuensi_tinggi")

    # 7. Normalisasi dengan frekuensi tinggi
    if decision == "normalisasi" and frequency >= 100:
        flags.append("normalisasi_frekuensi_tinggi")

    # 8. Saran normalisasi sama dengan kata awal
    if suggestion and suggestion.lower() == word:
        flags.append("saran_sama_dengan_kata_awal")

    # 9. Alasan menyebut nama entitas tetapi kata bukan
    # singkatan/nama yang jelas. Ini hanya indikator pemeriksaan.
    entity_terms = [
        "nama entitas",
        "nama orang",
        "nama tokoh"
    ]

    if any(term in reason for term in entity_terms):
        if category not in {"nama_entitas", "istilah_khusus"}:
            flags.append("cek_alasan_nama_entitas")

    # 10. Kategori tidak diketahui tetapi keputusan dipertahankan
    if category == "tidak_diketahui" and decision == "pertahankan":
        flags.append("kategori_tidak_diketahui_dipertahankan")

    return flags

# PROSES AUDIT
def main():

    print("=" * 60)
    print("AUDIT HASIL KLASIFIKASI KOSAKATA")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "word",
        "frequency",
        "examples",
        "category",
        "suggested_normalization",
        "decision",
        "confidence",
        "reason"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom tidak ditemukan: {missing_columns}"
        )

    # Bersihkan nilai teks
    for col in [
        "word",
        "category",
        "suggested_normalization",
        "decision",
        "confidence",
        "reason"
    ]:
        df[col] = df[col].apply(clean_text)

    df["frequency"] = pd.to_numeric(
        df["frequency"],
        errors="coerce"
    ).fillna(0).astype(int)

    # Ubah contoh menjadi teks yang mudah dibaca
    df["examples_readable"] = df["examples"].apply(
        parse_examples
    )

    # Jalankan audit setiap baris
    df["audit_flags"] = df.apply(
        audit_row,
        axis=1
    )

    df["audit_status"] = df["audit_flags"].apply(
        lambda x: "perlu_diperiksa" if x else "tidak_terindikasi"
    )

    df["audit_flags"] = df["audit_flags"].apply(
        lambda x: "; ".join(x)
    )

    # Urutkan berdasarkan prioritas pemeriksaan
    df = df.sort_values(
        by=["audit_status", "frequency"],
        ascending=[True, False]
    )

    # SIMPAN HASIL
    all_file = OUTPUT_DIR / "audit_dictionary_all.csv"

    df.to_csv(
        all_file,
        index=False,
        encoding="utf-8-sig"
    )

    # Semua baris yang memiliki indikator masalah
    flagged = df[
        df["audit_status"] == "perlu_diperiksa"
    ].copy()

    flagged_file = OUTPUT_DIR / "audit_dictionary_flagged.csv"

    flagged.to_csv(
        flagged_file,
        index=False,
        encoding="utf-8-sig"
    )

    # Semua kata yang disarankan untuk dinormalisasi
    normalization = df[
        df["decision"].str.lower() == "normalisasi"
    ].copy()

    normalization = normalization.sort_values(
        "frequency",
        ascending=False
    )

    normalization_file = (
        OUTPUT_DIR / "audit_normalisasi.csv"
    )

    normalization.to_csv(
        normalization_file,
        index=False,
        encoding="utf-8-sig"
    )

    # Semua kata yang masih perlu review manual
    manual = df[
        df["decision"].str.lower() == "review_manual"
    ].copy()

    manual = manual.sort_values(
        "frequency",
        ascending=False
    )

    manual_file = OUTPUT_DIR / "audit_review_manual.csv"

    manual.to_csv(
        manual_file,
        index=False,
        encoding="utf-8-sig"
    )

    # Ringkasan indikator
    summary = {
        "total_kosakata": len(df),
        "terindikasi_perlu_diperiksa": len(flagged),
        "tidak_terindikasi": len(df) - len(flagged),
        "total_normalisasi": len(normalization),
        "total_review_manual": len(manual),
        "jumlah_kata_umum_review_manual": int(
            df["audit_flags"].str.contains(
                "kata_umum_review_manual",
                regex=False
            ).sum()
        ),
        "jumlah_review_manual_frekuensi_tinggi": int(
            df["audit_flags"].str.contains(
                "review_manual_frekuensi_tinggi",
                regex=False
            ).sum()
        )
    }

    summary_file = OUTPUT_DIR / "audit_summary.txt"

    with open(summary_file, "w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    # ========================================================
    # TAMPILKAN RINGKASAN
    # ========================================================

    print("\nAUDIT SELESAI")
    print("-" * 60)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nFile hasil:")
    print(all_file)
    print(flagged_file)
    print(normalization_file)
    print(manual_file)
    print(summary_file)


if __name__ == "__main__":
    main()