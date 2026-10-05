"""
File ini digunakan untuk melakukan audit awal terhadap dataset hasil scraping
sebelum dilakukan filtering, preprocessing, dan labeling.

Pemeriksaan yang dilakukan:
- Jumlah seluruh komentar hasil scraping
- Jumlah comment_id unik
- Duplikasi comment_id
- Missing value pada kolom penting
- Komentar kosong
- Jumlah video sumber
- Memastikan video yang dikecualikan tidak masuk dataset
- Distribusi komentar per video
- Jumlah top-level comment dan reply
- Rentang tanggal komentar
- Statistik panjang komentar

Input:
    data/raw/comments_raw.csv

Output:
    Ringkasan audit ditampilkan di terminal.

Tujuan:
    Mengetahui kondisi awal (raw) dataset sebelum masuk ke tahap filtering
    komentar berbahasa Indonesia dan pembersihan data lainnya.
"""

import pandas as pd


# File dataset hasil scraping
INPUT_FILE = "data/raw/comments_raw.csv"

# Video 44 detik yang sebelumnya dikecualikan
EXCLUDED_VIDEO_ID = "wfdK1bl386A"


def main():
    print("=" * 70)
    print("RAW DATASET AUDIT")
    print("=" * 70)

    # Load dataset
    df = pd.read_csv(INPUT_FILE)

    # INFORMASI DASAR DATASET

    print("\n[1] INFORMASI DASAR")
    print("-" * 70)

    print(f"Jumlah baris            : {len(df):,}")
    print(f"Jumlah kolom            : {len(df.columns)}")
    print(f"Jumlah comment_id unik  : {df['comment_id'].nunique():,}")
    print(f"Jumlah video unik       : {df['video_id'].nunique():,}")

    print("\nKolom dataset:")
    for column in df.columns:
        print(f"- {column}")

    # DUPLIKASI COMMENT ID

    print("\n[2] DUPLIKASI COMMENT ID")
    print("-" * 70)

    duplicate_count = df["comment_id"].duplicated().sum()

    print(f"Duplicate comment_id    : {duplicate_count:,}")

    if duplicate_count == 0:
        print("Status                  : Tidak ditemukan duplikasi comment_id")
    else:
        print("Status                  : Ditemukan duplikasi comment_id")

    # Detail duplicate comment_id
    if duplicate_count > 0:
        duplicate_rows = df[
            df.duplicated(
                subset=["comment_id"],
                keep=False
            )
        ].sort_values("comment_id")

        print(f"Total baris terkait duplicate : {len(duplicate_rows):,}")

        same_text = (
            duplicate_rows
            .groupby("comment_id")["comment"]
            .nunique(dropna=False)
        )

        different_text_count = (same_text > 1).sum()

        print(
            f"Duplicate dengan teks berbeda : "
            f"{different_text_count:,}"
        )

        print("\nContoh duplicate:")

        print(
            duplicate_rows[
                [
                    "comment_id",
                    "video_id",
                    "comment",
                    "is_reply"
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    # MISSING VALUES

    print("\n[3] MISSING VALUES")
    print("-" * 70)

    important_columns = [
        "comment_id",
        "video_id",
        "video_title",
        "username",
        "comment",
        "published_at"
    ]

    for column in important_columns:
        if column in df.columns:
            missing = df[column].isna().sum()
            print(f"{column:<20}: {missing:,}")

    # KOMENTAR KOSONG

    print("\n[4] KOMENTAR KOSONG")
    print("-" * 70)

    empty_comments = (
        df["comment"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(f"Komentar kosong         : {empty_comments:,}")

    # VALIDASI VIDEO YANG DIKECUALIKAN

    print("\n[5] VALIDASI VIDEO EXCLUSION")
    print("-" * 70)

    excluded_count = (
        df["video_id"]
        .astype(str)
        .eq(EXCLUDED_VIDEO_ID)
        .sum()
    )

    print(f"Video ID                : {EXCLUDED_VIDEO_ID}")
    print(f"Komentar ditemukan      : {excluded_count:,}")

    if excluded_count == 0:
        print("Status                  : OK - video tidak masuk dataset")
    else:
        print("Status                  : WARNING - video ditemukan dalam dataset")

    # ---------------------------------------------------------
    # 6. TOP-LEVEL COMMENT VS REPLY
    # ---------------------------------------------------------

    print("\n[6] TIPE KOMENTAR")
    print("-" * 70)

    if "is_reply" in df.columns:

        reply_status = (
            df["is_reply"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        reply_count = reply_status.eq("true").sum()
        top_level_count = reply_status.eq("false").sum()
        unknown_count = len(df) - reply_count - top_level_count

        print(f"Top-level comment       : {top_level_count:,}")
        print(f"Reply                   : {reply_count:,}")

        if unknown_count > 0:
            print(f"Tipe tidak dikenali     : {unknown_count:,}")

    # DISTRIBUSI KOMENTAR PER VIDEO

    print("\n[7] DISTRIBUSI KOMENTAR PER VIDEO")
    print("-" * 70)

    video_distribution = (
        df.groupby(["video_id", "video_title"])
        .size()
        .reset_index(name="total_comments")
        .sort_values(
            by="total_comments",
            ascending=False
        )
    )

    for _, row in video_distribution.iterrows():
        print(
            f"{row['total_comments']:>7,} | "
            f"{row['video_title']}"
        )

    # RENTANG TANGGAL KOMENTAR

    print("\n[8] RENTANG TANGGAL KOMENTAR")
    print("-" * 70)

    published_dates = pd.to_datetime(
        df["published_at"],
        errors="coerce",
        utc=True
    )

    valid_dates = published_dates.dropna()

    if not valid_dates.empty:
        print(
            "Komentar paling awal    : "
            f"{valid_dates.min()}"
        )

        print(
            "Komentar paling akhir   : "
            f"{valid_dates.max()}"
        )
    else:
        print("Tidak ada tanggal valid.")

    invalid_dates = published_dates.isna().sum()

    print(f"Tanggal tidak valid     : {invalid_dates:,}")

    # STATISTIK PANJANG KOMENTAR

    print("\n[9] PANJANG KOMENTAR")
    print("-" * 70)

    comment_length = (
        df["comment"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    print(f"Minimum karakter        : {comment_length.min():,}")
    print(f"Maximum karakter        : {comment_length.max():,}")
    print(f"Rata-rata karakter      : {comment_length.mean():.2f}")
    print(f"Median karakter         : {comment_length.median():.2f}")

    # SUMMARY

    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    print(f"Raw comments            : {len(df):,}")
    print(f"Unique comment IDs      : {df['comment_id'].nunique():,}")
    print(f"Duplicate IDs           : {duplicate_count:,}")
    print(f"Empty comments          : {empty_comments:,}")
    print(f"Unique videos           : {df['video_id'].nunique():,}")
    print(f"Excluded video comments : {excluded_count:,}")

    print("=" * 70)
    print("Audit selesai. Dataset raw tidak diubah.")


if __name__ == "__main__":
    main()