# Menggabungkan kamus normalisasi v3 dengan kandidat kamus alay.
# Memprioritaskan keputusan kamus v3 jika terdapat duplikasi.
# Memeriksa konflik dan mencatat kandidat yang tidak digabungkan.
# Tidak mengubah dataset komentar atau file kamus sebelumnya.

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

AUDIT_DIR = ROOT / "data" / "processed" / "vocabulary_audit"
V3_PATH = AUDIT_DIR / "normalization_dictionary_v3.csv"

ALAY_PATH = (
    AUDIT_DIR / "kamusalay_extension"
    / "alay_candidates_priority_50.csv"
)

OUTPUT_DIR = AUDIT_DIR / "kamusalay_extension"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "normalization_dictionary_merged.csv"
REPORT_PATH = OUTPUT_DIR / "merge_report.csv"

# Pasangan hasil seleksi kandidat kamus alay.
# Target disesuaikan dengan kebijakan normalisasi penelitian.
APPROVED = {
    "ga": "tidak",
    "gak": "tidak",
    "gk": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "aja": "saja",
    "udah": "sudah",
    "bgt": "banget",
    "klo": "kalau",
    "tp": "tetapi",
    "emang": "memang",
    "jg": "juga",
    "jd": "jadi",
    "krn": "karena",
    "krna": "karena",
    "org": "orang",
    "gitu": "begitu",
    "nonton": "menonton",
    "gimana": "bagaimana",
    "gini": "begini",
    "nunggu": "menunggu",
    "nungguin": "menunggu",
    "trus": "terus",
    "ngikutin": "mengikuti",
    "kasian": "kasihan",
    "lg": "lagi",
    "pake": "pakai",
    "dg": "dengan",
    "tetep": "tetap",
    "skrg": "sekarang",
    "ngaku": "mengaku",
    "bs": "bisa",
    "jgn": "jangan",
    "tu": "itu",
    "tdk": "tidak",
    "bener2": "benar-benar",
    "keliatan": "kelihatan",
    "ttg": "tentang",
}

# Kata yang tidak boleh dimasukkan secara otomatis.
BLOCKED = {
    "ksh", "si", "minta", "gw",
    "fans", "beneran", "so", "ngomong",
    "kaya", "ama", "kyk", "indo",
    "dah", "kek", "team", "kalaupun",
    "ajaaa", "yaaaa", "akhirnyaa", "akhirnyaaa",
    "iniii", "jugaa", "bgttt",
}


def main():
    if not V3_PATH.exists():
        raise FileNotFoundError(f"Kamus v3 tidak ditemukan: {V3_PATH}")

    if not ALAY_PATH.exists():
        raise FileNotFoundError(f"Kandidat alay tidak ditemukan: {ALAY_PATH}")

    v3 = pd.read_csv(V3_PATH, dtype=str, encoding="utf-8-sig")
    alay = pd.read_csv(ALAY_PATH, dtype=str, encoding="utf-8-sig")

    v3.columns = v3.columns.str.strip().str.lower()
    alay.columns = alay.columns.str.strip().str.lower()

    if not {"word", "normalisasi"}.issubset(v3.columns):
        raise ValueError(
            f"Kamus v3 harus memiliki kolom word dan normalisasi: {v3.columns.tolist()}"
        )

    if "slang" not in alay.columns:
        raise ValueError("File kandidat harus memiliki kolom slang.")

    v3["word"] = v3["word"].fillna("").str.strip().str.lower()
    v3["normalisasi"] = v3["normalisasi"].fillna("").str.strip().str.lower()

    if v3["word"].eq("").any() or v3["normalisasi"].eq("").any():
        raise ValueError("Kamus v3 memiliki kata atau normalisasi kosong.")

    if v3["word"].duplicated().any():
        raise ValueError("Kamus v3 memiliki kata sumber duplikat.")

    alay["slang"] = alay["slang"].fillna("").str.strip().str.lower()
    available = set(alay["slang"])

    # Mempertahankan semua keputusan dari kamus utama.
    merged = {
        row["word"]: row["normalisasi"]
        for _, row in v3.iterrows()
    }

    records = []
    report = []

    for _, row in v3.iterrows():
        records.append({
            "word": row["word"],
            "normalisasi": row["normalisasi"],
            "sumber": "kamus_v3",
            "status": "kamus_utama"
        })

    for word, target in APPROVED.items():
        if word in BLOCKED:
            action = "ditolak_blocklist"

        elif word not in available:
            action = "tidak_ada_di_prioritas_50"

        elif word in merged:
            if merged[word] == target:
                action = "duplikat_sama"
            else:
                action = "konflik_prioritas_v3"

        else:
            merged[word] = target

            records.append({
                "word": word,
                "normalisasi": target,
                "sumber": "Salsabila_et_al_2018",
                "status": "tambahan_terseleksi"
            })

            action = "ditambahkan"

        report.append({
            "word": word,
            "usulan_normalisasi": target,
            "keputusan": action,
            "normalisasi_final": merged.get(word, "")
        })

    output = pd.DataFrame(records)
    report_df = pd.DataFrame(report)

    output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    report_df.to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")

    print("Penggabungan kamus selesai.")
    print(f"Kamus utama: {len(v3)} pasangan")
    print(f"Kandidat yang diperiksa: {len(APPROVED)}")
    print(
        "Kandidat ditambahkan:",
        (report_df["keputusan"] == "ditambahkan").sum()
    )
    print(
        "Konflik:",
        (report_df["keputusan"] == "konflik_prioritas_v3").sum()
    )
    print(
        "Kamus gabungan:",
        len(output)
    )
    print(f"Hasil: {OUTPUT_PATH}")
    print(f"Laporan: {REPORT_PATH}")


if __name__ == "__main__":
    main()