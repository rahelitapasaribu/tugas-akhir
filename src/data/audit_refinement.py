"""
inspect comment flagged for language review
"""
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "interim"
input_path = DATA_DIR / "comments_refined.csv"
output_path = DATA_DIR / "review_sample.csv"

def main():
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    
    # select comments flagged for manual review
    review_df = df[
        df["language_candidate"] == "review"
    ].copy()
    print(f"Total review comments: {len(review_df)}")
    
    # select up to 100 comments for an intiial audit
    sample_size = min(100, len(review_df))
    sample_df = review_df.sample(
        n=sample_size, 
        random_state=42
    )
    
    # keep columns needed for manual inspection
    columns = [
        "comment_id",
        "comment",
        "detected_language",
        "language_candidate",
    ]
    sample_df[columns].to_csv(
        output_path, 
        index=False,
        encoding="utf-8-sig"
    )
    print(f"Audit sample saved to: {output_path}")
    print(f"Sample size: {len(sample_df)}")
    
if __name__ == "__main__":
    main()
