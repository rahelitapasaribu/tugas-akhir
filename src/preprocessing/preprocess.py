
# Melakukan case folding dan cleaning komentar pada seluruh dataset train, validation, dan test.

import re
from pathlib import Path
import pandas as pd
import emoji

# Lokasi folder
INPUT_DIR = Path("data/processed/splits_custom")
OUTPUT_DIR = Path("data/processed/splits_preprocessed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Emoji yang dipertahankan karena berpotensi menyampaikan sentimen.
SENTIMENT_EMOJIS = {
    "😂", "🤣", "😭", "😢", "🥺", "😔", "😞", "😡", "🤬",
    "😍", "🥰", "❤️", "❤", "💕", "💖", "💗", "💔",
    "😠", "😤", "😩", "😫", "😱", "🤔", "😥", "😓",
    "😊", "☺️", "😁", "😄", "😃", "😆", "😘", "👏",
    "🙏", "👍", "👎", "🙄", "💀", "🫶"
}

def clean_comment(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Case folding
    text = text.lower()

    # Menghapus URL
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Menghapus emoji yang tidak termasuk daftar emoji sentimen
    for item in emoji.emoji_list(text):
        emoji_text = item["emoji"]
        if emoji_text not in SENTIMENT_EMOJIS:
            text = text.replace(emoji_text, " ")

    # Merapikan spasi
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Memproses setiap split
for split_name in ["train", "validation", "test"]:
    input_path = INPUT_DIR / f"{split_name}.csv"
    output_path = OUTPUT_DIR / f"{split_name}.csv"

    df = pd.read_csv(input_path)

    if "comment" not in df.columns:
        raise ValueError(f"Kolom 'comment' tidak ditemukan pada {input_path}")

    # Membuat kolom baru tanpa mengubah komentar asli
    df["comment_clean"] = df["comment"].apply(clean_comment)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"{split_name.upper()}: {len(df)} data berhasil diproses.")
    print(f"Disimpan di: {output_path}")

print("\nPreprocessing selesai.")