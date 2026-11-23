"""
FINALISASI LABEL SENTIMEN

Kegunaan:
1. Membaca dataset 10.000 komentar yang sudah selesai preprocessing.
2. Membaca hasil audit 150 sampel dan master keputusan label.
3. Menerapkan keputusan manual/audit yang tercatat di master keputusan.
4. Mempertahankan label Qwen untuk komentar yang tidak memiliki keputusan koreksi.
5. Membuat kolom label_final dan sumber label agar proses dapat ditelusuri.
6. Memvalidasi ID, konflik keputusan, dan label sebelum menyimpan output.
7. Menyimpan dataset final, distribusi label, dan laporan finalisasi.

Script ini TIDAK melakukan preprocessing ulang dan TIDAK menghapus komentar spam,
promosi, atau off-topic.

Prioritas label:
- Prioritas 1: label manual pengguna.
- Prioritas 2: keputusan batch yang disetujui.
- Prioritas 3: label yang dipertahankan dari review Batch 4.
- Jika tidak ada keputusan di master, label Qwen dipertahankan.

Jalankan dari root proyek tugas-akhir:
    python src/labeling/finalize_labels.py
"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREPROCESSED_FILE = (
    PROJECT_ROOT / "data" / "processed" / "comments_preprocessed_final.csv"
)
AUDIT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "label_audit"
    / "label_review_150_gabungan.csv"
)
MASTER_FILE = (
    PROJECT_ROOT / "data" / "processed" / "label_audit"
    / "master_keputusan_label.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "comments_labeled_final.csv"
)
DISTRIBUTION_FILE = (
    PROJECT_ROOT / "results" / "metrics" / "label_final_distribution.csv"
)
REPORT_FILE = (
    PROJECT_ROOT / "results" / "metrics" / "label_finalization_report.csv"
)

VALID_LABELS = {"Positif", "Negatif", "Netral"}


def normalize_id(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_label(value):
    if pd.isna(value):
        return ""
    value = str(value).strip().lower()
    mapping = {
        "positif": "Positif",
        "positive": "Positif",
        "pos": "Positif",
        "negatif": "Negatif",
        "negative": "Negatif",
        "neg": "Negatif",
        "netral": "Netral",
        "neutral": "Netral",
        "neu": "Netral",
    }
    return mapping.get(value, "")


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")


def require_columns(df, columns, file_description):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan pada {file_description}: {missing}\n"
            f"Kolom tersedia: {list(df.columns)}"
        )


def main():
    print("Mulai finalisasi label...")

    for path in (PREPROCESSED_FILE, AUDIT_FILE, MASTER_FILE):
        require_file(path)

    comments = pd.read_csv(PREPROCESSED_FILE, dtype={"comment_id": str})
    audit = pd.read_csv(AUDIT_FILE, dtype={"comment_id": str})
    master = pd.read_csv(MASTER_FILE, dtype={"comment_id": str})

    require_columns(
        comments,
        ["comment_id", "sentiment_qwen"],
        "dataset preprocessing",
    )
    require_columns(
        audit,
        ["nomor_sampel", "comment_id"],
        "file audit 150",
    )
    require_columns(
        master,
        [
            "nomor_sampel",
            "comment_id",
            "label_keputusan",
            "jenis_keputusan",
            "prioritas",
        ],
        "master keputusan",
    )

    comments["comment_id"] = comments["comment_id"].map(normalize_id)
    audit["comment_id"] = audit["comment_id"].map(normalize_id)
    master["comment_id"] = master["comment_id"].map(normalize_id)

    comments["_qwen_label"] = comments["sentiment_qwen"].map(normalize_label)

    if comments["comment_id"].eq("").any():
        raise ValueError("Ada comment_id kosong di dataset utama.")
    if comments["comment_id"].duplicated().any():
        raise ValueError("comment_id duplikat ditemukan di dataset utama.")
    if not comments["_qwen_label"].isin(VALID_LABELS).all():
        bad = comments.loc[
            ~comments["_qwen_label"].isin(VALID_LABELS),
            ["comment_id", "sentiment_qwen"],
        ]
        raise ValueError(
            "Label Qwen invalid ditemukan. Contoh:\n"
            + bad.head(10).to_string(index=False)
        )

    # Membuat peta nomor sampel audit ke comment_id.
    audit["_sample_number"] = pd.to_numeric(
        audit["nomor_sampel"], errors="coerce"
    )
    if audit["_sample_number"].isna().any():
        raise ValueError("Ada nomor_sampel audit yang tidak dapat dibaca.")

    sample_to_comment = audit.set_index("_sample_number")["comment_id"].to_dict()

    master["_sample_number"] = pd.to_numeric(
        master["nomor_sampel"], errors="coerce"
    )
    master["label_keputusan_normalized"] = master["label_keputusan"].map(
        normalize_label
    )
    master["prioritas_numeric"] = pd.to_numeric(
        master["prioritas"], errors="coerce"
    )

    if master["_sample_number"].isna().any():
        raise ValueError("Ada nomor_sampel pada master yang tidak valid.")
    if master["prioritas_numeric"].isna().any():
        raise ValueError("Ada nilai prioritas pada master yang tidak valid.")
    if not master["label_keputusan_normalized"].isin(VALID_LABELS).all():
        raise ValueError("Ada label_keputusan master yang tidak valid.")

    # Gunakan comment_id di master jika tersedia; jika kosong, cari dari nomor sampel audit.
    master["_resolved_comment_id"] = master["comment_id"]
    missing_master_id = master["_resolved_comment_id"].eq("")
    master.loc[missing_master_id, "_resolved_comment_id"] = (
        master.loc[missing_master_id, "_sample_number"].map(sample_to_comment)
    )

    unresolved = master[master["_resolved_comment_id"].isna()]
    if not unresolved.empty:
        raise ValueError(
            "Ada keputusan master yang tidak dapat dipetakan ke comment_id:\n"
            + unresolved[
                ["nomor_sampel", "label_keputusan", "jenis_keputusan"]
            ].to_string(index=False)
        )

    # Pastikan seluruh keputusan master merujuk komentar dalam dataset 10.000.
    main_ids = set(comments["comment_id"])
    master_not_in_main = master[
        ~master["_resolved_comment_id"].isin(main_ids)
    ]
    if not master_not_in_main.empty:
        raise ValueError(
            "Ada keputusan master yang comment_id-nya tidak ditemukan "
            "di dataset 10.000:\n"
            + master_not_in_main[
                ["nomor_sampel", "_resolved_comment_id", "label_keputusan"]
            ].to_string(index=False)
        )

    # Urutkan menurut prioritas. Bila keputusan pada prioritas tertinggi
    # untuk komentar yang sama berbeda label, proses dihentikan untuk pemeriksaan.
    decision_rows = master[
        [
            "_resolved_comment_id",
            "label_keputusan_normalized",
            "jenis_keputusan",
            "prioritas_numeric",
            "nomor_sampel",
        ]
    ].copy()

    resolved_decisions = []
    conflict_rows = []

    for comment_id, group in decision_rows.groupby("_resolved_comment_id"):
        highest_priority = group["prioritas_numeric"].min()
        top = group[group["prioritas_numeric"] == highest_priority]
        labels_at_top = top["label_keputusan_normalized"].unique()

        if len(labels_at_top) > 1:
            conflict_rows.append({
                "comment_id": comment_id,
                "prioritas": highest_priority,
                "label_conflict": "|".join(sorted(labels_at_top)),
                "nomor_sampel": "|".join(
                    top["nomor_sampel"].astype(str).tolist()
                ),
            })
            continue

        chosen = top.iloc[0]
        resolved_decisions.append({
            "comment_id": comment_id,
            "label_final_override": chosen["label_keputusan_normalized"],
            "label_source": chosen["jenis_keputusan"],
            "decision_priority": int(highest_priority),
        })

    if conflict_rows:
        conflict_df = pd.DataFrame(conflict_rows)
        conflict_path = OUTPUT_FILE.parent / "label_finalization_conflicts.csv"
        conflict_df.to_csv(conflict_path, index=False, encoding="utf-8-sig")
        raise ValueError(
            "Konflik keputusan pada prioritas tertinggi ditemukan. "
            f"Rincian disimpan di: {conflict_path}"
        )

    decisions = pd.DataFrame(resolved_decisions)
    if decisions["comment_id"].duplicated().any():
        raise ValueError("Masih ada comment_id keputusan duplikat setelah resolusi.")

    # Terapkan keputusan yang sudah direkonsiliasi; lainnya memakai label Qwen.
    final = comments.merge(
        decisions,
        on="comment_id",
        how="left",
        validate="one_to_one",
    )

    final["label_final"] = final["label_final_override"].where(
        final["label_final_override"].isin(VALID_LABELS),
        final["_qwen_label"],
    )
    final["label_source"] = final["label_source"].fillna("Qwen_dipertahankan")

    if not final["label_final"].isin(VALID_LABELS).all():
        raise ValueError("Ada label_final kosong atau tidak valid.")

    # Ringkasan distribusi label final.
    distribution = (
        final["label_final"]
        .value_counts()
        .reindex(["Positif", "Negatif", "Netral"], fill_value=0)
        .rename_axis("label")
        .reset_index(name="jumlah")
    )
    distribution["persentase"] = (
        distribution["jumlah"] / len(final) * 100
    ).round(2)

    # Ringkasan sumber label dan pengecekan jumlah.
    source_distribution = (
        final["label_source"]
        .value_counts()
        .rename_axis("label_source")
        .reset_index(name="jumlah")
    )

    batch1_count = int(master["jenis_keputusan"].eq("batch_1_disetujui").sum())
    report = pd.DataFrame([{
        "jumlah_data_input": len(comments),
        "jumlah_data_output": len(final),
        "jumlah_keputusan_master": len(master),
        "batch_1_target": 25,
        "batch_1_tercatat": batch1_count,
        "batch_1_tidak_tercatat": max(25 - batch1_count, 0),
        "catatan_batch_1": (
            "1 keputusan Batch 1 belum tercatat; sampel tersebut memakai label Qwen "
            "jika tidak ada keputusan lain di master."
            if batch1_count < 25 else "Batch 1 lengkap."
        ),
        "jumlah_comment_id_keputusan_unik": len(decisions),
        "jumlah_label_qwen_dipertahankan": int(
            (final["label_source"] == "Qwen_dipertahankan").sum()
        ),
        "jumlah_label_override": int(
            (final["label_source"] != "Qwen_dipertahankan").sum()
        ),
        "jumlah_label_final_invalid": int(
            (~final["label_final"].isin(VALID_LABELS)).sum()
        ),
        "status": "berhasil",
    }])

    # Simpan hasil.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    DISTRIBUTION_FILE.parent.mkdir(parents=True, exist_ok=True)

    final.drop(
        columns=[
            "_qwen_label",
            "label_final_override",
            "decision_priority",
        ],
        errors="ignore",
    ).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    distribution.to_csv(
        DISTRIBUTION_FILE, index=False, encoding="utf-8-sig"
    )
    report.to_csv(
        REPORT_FILE, index=False, encoding="utf-8-sig"
    )

    print("\nFinalisasi label selesai.")
    print(f"Dataset final: {OUTPUT_FILE}")
    print(f"Distribusi: {DISTRIBUTION_FILE}")
    print(f"Laporan: {REPORT_FILE}")
    print("\nDistribusi label:")
    print(distribution.to_string(index=False))
    print("\nSumber label:")
    print(source_distribution.to_string(index=False))
    print("\nLaporan:")
    print(report.to_string(index=False))
    if batch1_count < 25:
        print(
            "\\nPERINGATAN: Batch 1 baru tercatat "
            f"{batch1_count}/25 keputusan. Keputusan yang tidak tercatat "
            "tidak ditebak; label Qwen dipakai sebagai fallback."
        )

if __name__ == "__main__":
    main()