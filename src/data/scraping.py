# scraping tanggal 5 Oktober 2026

import os
import time

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# Load environment variables
load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    raise ValueError("YOUTUBE_API_KEY tidak ditemukan di file .env")


# File paths
PLAYLIST_FILE = "data/raw/playlist_videos.csv"
OUTPUT_FILE = "data/raw/comments_raw.csv"


# Video yang tidak digunakan dalam penelitian
EXCLUDED_VIDEO_IDS = {
    "wfdK1bl386A"  # Update-an (00:44)
}


# YouTube API client
youtube = build(
    "youtube",
    "v3",
    developerKey=API_KEY
)


def get_replies(parent_id, video_id, video_title):
    # Ambil seluruh reply dari sebuah top-level comment
    replies = []
    next_page_token = None

    while True:
        request = youtube.comments().list(
            part="snippet",
            parentId=parent_id,
            maxResults=100,
            pageToken=next_page_token,
            textFormat="plainText"
        )

        response = request.execute()

        for item in response.get("items", []):
            snippet = item["snippet"]

            replies.append({
                "comment_id": item["id"],
                "video_id": video_id,
                "video_title": video_title,
                "username": snippet.get("authorDisplayName"),
                "comment": snippet.get("textDisplay"),
                "like_count": snippet.get("likeCount", 0),
                "published_at": snippet.get("publishedAt"),
                "updated_at": snippet.get("updatedAt"),
                "reply_count": 0,
                "is_reply": True,
                "parent_id": parent_id
            })

        next_page_token = response.get("nextPageToken")

        if not next_page_token:
            break

    return replies


def get_comments(video_id, video_title):
    # Ambil seluruh top-level comment dan reply dari satu video
    comments = []
    next_page_token = None

    top_level_count = 0
    reply_count_total = 0

    while True:
        request = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
            textFormat="plainText",
            order="time"
        )

        response = request.execute()

        for item in response.get("items", []):
            thread_snippet = item["snippet"]
            top_comment = thread_snippet["topLevelComment"]
            snippet = top_comment["snippet"]

            parent_id = top_comment["id"]
            total_reply_count = thread_snippet.get("totalReplyCount", 0)

            comments.append({
                "comment_id": parent_id,
                "video_id": video_id,
                "video_title": video_title,
                "username": snippet.get("authorDisplayName"),
                "comment": snippet.get("textDisplay"),
                "like_count": snippet.get("likeCount", 0),
                "published_at": snippet.get("publishedAt"),
                "updated_at": snippet.get("updatedAt"),
                "reply_count": total_reply_count,
                "is_reply": False,
                "parent_id": None
            })

            top_level_count += 1

            # Ambil reply secara terpisah agar semua reply didapatkan
            if total_reply_count > 0:
                replies = get_replies(
                    parent_id,
                    video_id,
                    video_title
                )

                comments.extend(replies)
                reply_count_total += len(replies)

        next_page_token = response.get("nextPageToken")

        if not next_page_token:
            break

    return comments, top_level_count, reply_count_total


def save_progress(comments):
    # Simpan progres agar data tidak hilang jika proses berhenti
    if not comments:
        return

    df = pd.DataFrame(comments)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )


def main():
    if not os.path.exists(PLAYLIST_FILE):
        raise FileNotFoundError(
            f"{PLAYLIST_FILE} tidak ditemukan. "
            "Jalankan get_playlist_videos.py terlebih dahulu."
        )

    playlist_df = pd.read_csv(PLAYLIST_FILE)

    required_columns = {"video_id", "title"}

    if not required_columns.issubset(playlist_df.columns):
        raise ValueError(
            "playlist_videos.csv harus memiliki kolom video_id dan title."
        )

    # Exclude video yang tidak digunakan
    videos_df = playlist_df[
        ~playlist_df["video_id"].isin(EXCLUDED_VIDEO_IDS)
    ].copy()

    print("=" * 70)
    print("YOUTUBE COMMENT SCRAPING")
    print("=" * 70)
    print(f"Video dalam playlist : {len(playlist_df)}")
    print(f"Video dikecualikan   : {len(playlist_df) - len(videos_df)}")
    print(f"Video akan diproses  : {len(videos_df)}")
    print("=" * 70)

    all_comments = []

    total_top_level = 0
    total_replies = 0

    for index, row in enumerate(videos_df.itertuples(index=False), start=1):
        video_id = row.video_id
        video_title = row.title

        print(f"\n[{index}/{len(videos_df)}]")
        print(f"Judul    : {video_title}")
        print(f"Video ID : {video_id}")
        print("Mengambil komentar...")

        try:
            comments, top_count, reply_count = get_comments(
                video_id,
                video_title
            )

            all_comments.extend(comments)

            total_top_level += top_count
            total_replies += reply_count

            print(f"Top-level : {top_count}")
            print(f"Replies   : {reply_count}")
            print(f"Total     : {len(comments)}")

            # Simpan setiap selesai satu video
            save_progress(all_comments)

            # Jeda kecil antarvideo
            time.sleep(0.5)

        except HttpError as error:
            status_code = getattr(error.resp, "status", None)

            print(f"Gagal mengambil komentar: HTTP {status_code}")
            print(error)

            # Tetap simpan data yang sudah berhasil diperoleh
            save_progress(all_comments)

            if status_code == 403:
                print(
                    "Kemungkinan komentar dinonaktifkan, "
                    "akses dibatasi, atau quota API bermasalah."
                )

            print("Melanjutkan ke video berikutnya...")

    # Final save
    save_progress(all_comments)

    print("\n" + "=" * 70)
    print("SCRAPING SELESAI")
    print("=" * 70)
    print(f"Video diproses          : {len(videos_df)}")
    print(f"Total top-level comment : {total_top_level}")
    print(f"Total replies           : {total_replies}")
    print(f"Total seluruh komentar  : {len(all_comments)}")
    print(f"File                     : {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()