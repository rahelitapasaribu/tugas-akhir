# Menggabungkan 48 keputusan manual ke audit 150 komentar.
# Mempertahankan label awal Qwen sebagai pembanding.
# Menandai 102 komentar lainnya sebagai belum divalidasi manual.
# Menghitung kesesuaian Qwen hanya pada 48 komentar tervalidasi.
# Tidak mengubah dataset preprocessing atau label Qwen asli.

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data" / "processed" / "label_audit"

MASTER_PATH = INPUT_DIR / "label_review_150_usulan_audit.csv"
PRIORITY_PATH = INPUT_DIR / "label_review_perlu_keputusan.csv"

OUTPUT_PATH = INPUT_DIR / "label_review_150_gabungan.csv"
SUMMARY_PATH = INPUT_DIR / "label_validation_summary.csv"
DIFFERENCES_PATH = INPUT_DIR / "label_qwen_disagreements.csv"

# Keputusan manual sesuai urutan 48 baris yang dikirim.
MANUAL_LABELS = """
Negatif
Negatif
Negatif
Netral
Netral
Positif
Netral
Netral
Netral
Negatif
Netral
Positif
Negatif
Netral
Positif
Netral
Netral
Netral
Negatif
Netral
Netral
Netral
Netral
Netral
Netral
Netral
Netral
Netral
Netral
Negatif
Positif
Netral
Positif
Negatif
Negatif
Negatif
Negatif
Negatif
Positif
Positif
Negatif
Netral
Positif
Negatif
Positif
Positif
Netral
Positif
""".strip().splitlines()

# Menyeragamkan label menjadi Positif, Negatif, atau Netral.
LABEL_MAP = {
    "positif": "Positif",
    "negatif": "Negatif",
    "netral": "Netral"
}

def normalize_label(value):
    label = str(value).strip().lower()
    return LABEL_MAP.get(label, "")

def main():
    # Membaca file audit utama dan komentar prioritas.
    master = pd.read_csv(
        MASTER_PATH,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig"
    )

    priority = pd.read_csv(
        PRIORITY_PATH,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig"
    )

    # Memeriksa jumlah data.
    if len(master) != 150:
        raise ValueError(
            f"Audit utama seharusnya 150 baris, ditemukan {len(master)}."
        )

    if len(priority) != 48:
        raise ValueError(
            f"File prioritas seharusnya 48 baris, ditemukan {len(priority)}."
        )

    if len(MANUAL_LABELS) != 48:
        raise ValueError(
            f"Jumlah keputusan manual seharusnya 48, ditemukan {len(MANUAL_LABELS)}."
        )

    # Memastikan kolom yang diperlukan tersedia.
    required_columns = {"comment_id", "sentiment_qwen"}

    if not required_columns.issubset(master.columns):
        raise ValueError(
            f"Kolom audit utama tidak lengkap: {master.columns.tolist()}"
        )

    if "comment_id" not in priority.columns:
        raise ValueError(
            "Kolom comment_id tidak ditemukan pada file prioritas."
        )

    # Memastikan ID komentar unik.
    if master["comment_id"].duplicated().any():
        raise ValueError("Ada comment_id duplikat pada audit utama.")

    if priority["comment_id"].duplicated().any():
        raise ValueError("Ada comment_id duplikat pada file prioritas.")

    if not set(priority["comment_id"]).issubset(set(master["comment_id"])):
        raise ValueError(
            "Ada comment_id dari file prioritas yang tidak ditemukan di audit utama."
        )

    # Menormalisasi 48 label manual.
    manual = [normalize_label(x) for x in MANUAL_LABELS]
    
    if any(label == "" for label in manual):
        raise ValueError("Ada keputusan manual yang tidak valid.")

    # Memasangkan label manual dengan comment_id.
    # Urutan file prioritas harus sama dengan urutan jawaban pengguna.
    decisions = dict(zip(priority["comment_id"], manual))

    # Menggabungkan keputusan manual ke 150 sampel.
    master["label_manual"] = (
        master["comment_id"].map(decisions).fillna("")
    )

    validated_mask = master["label_manual"].ne("")

    # Menandai status validasi setiap komentar.
    master["status_validasi"] = "Belum divalidasi manual"

    master.loc[
        validated_mask,
        "status_validasi"
    ] = "Divalidasi pengguna"

    # Menyeragamkan label Qwen untuk perbandingan.
    master["label_qwen_baku"] = (
        master["sentiment_qwen"].apply(normalize_label)
    )

    # Membandingkan label Qwen dengan keputusan manual.
    master["kesesuaian_qwen"] = ""

    master.loc[validated_mask, "kesesuaian_qwen"] = (
        master.loc[validated_mask, "label_qwen_baku"]
        == master.loc[validated_mask, "label_manual"]
    ).map({
        True: "Sesuai",
        False: "Berbeda"
    })

    # Menyimpan 150 sampel gabungan.
    master.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # Mengambil hanya komentar yang divalidasi manual.
    validated = master[validated_mask].copy()

    # Mengambil komentar dengan label yang berbeda.
    disagreements = validated[
        validated["kesesuaian_qwen"] == "Berbeda"
    ].copy()

    disagreements.to_csv(
        DIFFERENCES_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # Menghitung kesesuaian label.
    same_count = int(
        (validated["kesesuaian_qwen"] == "Sesuai").sum()
    )

    different_count = len(disagreements)

    agreement = (
        same_count / len(validated) * 100
        if len(validated) > 0
        else 0
    )

    # Membuat laporan ringkasan validasi.
    summary = pd.DataFrame([
        {
            "metrik": "Total sampel audit",
            "nilai": len(master)
        },
        {
            "metrik": "Divalidasi manual",
            "nilai": len(validated)
        },
        {
            "metrik": "Belum divalidasi manual",
            "nilai": len(master) - len(validated)
        },
        {
            "metrik": "Label Qwen sesuai",
            "nilai": same_count
        },
        {
            "metrik": "Label Qwen berbeda",
            "nilai": different_count
        },
        {
            "metrik": "Persentase kesesuaian (%)",
            "nilai": round(agreement, 2)
        }
    ])

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # Menampilkan hasil di terminal.
    print("Penggabungan validasi label selesai.")
    print(f"Total sampel: {len(master)}")
    print(f"Divalidasi manual: {len(validated)}")
    print(f"Belum divalidasi manual: {len(master) - len(validated)}")
    print(f"Label Qwen sesuai: {same_count}")
    print(f"Label Qwen berbeda: {different_count}")
    print(f"Kesesuaian pada 48 sampel: {agreement:.2f}%")
    print(f"Hasil gabungan: {OUTPUT_PATH}")
    print(f"Laporan: {SUMMARY_PATH}")
    print(f"Perbedaan label: {DIFFERENCES_PATH}")

if __name__ == "__main__":
    main()