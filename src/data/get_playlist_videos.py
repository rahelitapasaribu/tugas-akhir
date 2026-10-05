import os
import re

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build


# Load environment variables
load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
PLAYLIST_ID = "PLmieDp1s11zS8BhUjahqiCcA1f2Lr3HQ8"

if not API_KEY:
    raise ValueError("YOUTUBE_API_KEY tidak ditemukan di file .env")


# YouTube API client
youtube = build(
    "youtube",
    "v3",
    developerKey=API_KEY
)


def parse_duration(duration):
    # Convert ISO 8601 duration to seconds
    pattern = re.compile(
        r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?"
    )

    match = pattern.fullmatch(duration)

    if not match:
        return 0

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)

    return hours * 3600 + minutes * 60 + seconds


def format_duration(total_seconds):
    # Convert seconds to HH:MM:SS or MM:SS
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return f"{minutes:02d}:{seconds:02d}"


def get_playlist_videos():
    videos = []
    next_page_token = None

    while True:
        request = youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=PLAYLIST_ID,
            maxResults=50,
            pageToken=next_page_token
        )

        response = request.execute()

        for item in response["items"]:
            video_id = item["contentDetails"]["videoId"]

            videos.append({
                "video_id": video_id,
                "title": item["snippet"]["title"],
                "published_at": item["contentDetails"]["videoPublishedAt"],
                "video_url": f"https://www.youtube.com/watch?v={video_id}"
            })

        next_page_token = response.get("nextPageToken")

        if not next_page_token:
            break

    return videos


def get_video_durations(videos):
    # YouTube videos.list accepts up to 50 video IDs per request
    for i in range(0, len(videos), 50):
        batch = videos[i:i + 50]
        video_ids = [video["video_id"] for video in batch]

        request = youtube.videos().list(
            part="contentDetails",
            id=",".join(video_ids)
        )

        response = request.execute()

        duration_map = {}

        for item in response["items"]:
            duration_seconds = parse_duration(
                item["contentDetails"]["duration"]
            )

            duration_map[item["id"]] = duration_seconds

        for video in batch:
            seconds = duration_map.get(video["video_id"])

            video["duration_seconds"] = seconds
            video["duration"] = (
                format_duration(seconds)
                if seconds is not None
                else None
            )

    return videos


def main():
    print("Mengambil daftar video dari playlist...")

    videos = get_playlist_videos()
    videos = get_video_durations(videos)

    df = pd.DataFrame(videos)

    output_path = "data/raw/playlist_videos.csv"

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDaftar video berhasil diambil.")
    print(f"Jumlah video: {len(df)}")
    print(f"File disimpan: {output_path}\n")

    print(
        df[
            [
                "title",
                "duration",
                "video_id"
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()