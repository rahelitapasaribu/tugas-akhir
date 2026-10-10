"""
Menghapus duplikasi dan komentar kosong lalu menyaring komentar berbahasa Indonesia
"""
from pathlib import Path
import fasttext
import pandas as pd

input_file = Path("data/raw/comments_raw.csv")
output_file = Path("data/interim/comments_filterlang.csv")
model_file = Path("models/lid.176.bin")

def detect_language(text, model):
    text = " ".join(text.splitlines())
    if not text.strip():
        return ("unknown", 0.0)
    labels, scores = model.predict(text)
    language = labels[0].replace("__label__", "")
    confidence = scores[0]
    return (language, confidence)

def main():
    if not input_file.exists():
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
    if not model_file.exists():
        raise FileNotFoundError(f"Model file '{model_file}' not found.")
    
    print("=" * 70)
    print("Language Filtering")
    print("=" * 70)
    
    # load dataset
    df = pd.read_csv(input_file, dtype={"comment_id": "string"})
    raw_count = len(df)
    
    # hapus duplikasi comment_id
    df = df.drop_duplicates(subset=["comment_id"], keep="first").copy()
    removed_count = raw_count - len(df)
    
    # hapus komentar kosong
    empty_comments = df["comment"].isna() | (df["comment"].astype(str).str.strip().eq(""))
    empty_removed = int(empty_comments.sum())
    df = df.loc[~empty_comments].copy()
    print(f"Raw comments count: {raw_count:,}")
    print(f"Removed duplicate comment_id: {removed_count:,}")
    print(f"Removed empty comments: {empty_removed:,}")
    print(f"Remaining comments count: {len(df):,}")
    
    # load model
    print("Loading fasttext model...")
    model = fasttext.load_model(str(model_file))
    
    # deteksi bahasa
    print("Detecting language...")
    predictions = [detect_language(text,model) for text in df["comment"]]
    df["detected_language"] = [item[0] for item in predictions]
    df["language_confidence"] = [item[1] for item in predictions]
    
    # simpan hasil tanpa filter bahasa
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    
    # summary
    print("top 5 detected languages:")
    print(df["detected_language"].value_counts().head(5).to_string())
    
    print("Language detection summary")
    print("-" * 70)
    print(f"Total comments after filtering: {len(df):,}")
    print(f"Comments detected as Indonesian: {len(df[df['detected_language'] == 'id']):,}")
    print(f"Comments detected as non-Indonesian: {len(df[df['detected_language'] != 'id']):,}")
    print(f"Saved to: {output_file}")
    
if __name__ == "__main__":
    main()