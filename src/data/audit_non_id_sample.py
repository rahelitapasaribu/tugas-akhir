"""
audit sample komentar non-id
"""
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "interim"
input_path = DATA_DIR / "comments_refined.csv"
output_path = DATA_DIR / "non_id_sample.csv"

def main():
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    
    # take comments that dont included as id candidates
    candidates = df[
        df["language_candidate"] == "candidate_non_id"
    ].copy()
    print(f"Total candidate_non_id: {len(candidates)}")
    
    # take 100 random samples
    sample_size = min(100, len(candidates))
    sample_df = candidates.sample(
        n=sample_size, 
        random_state=42
    ).copy()
    
    # column for manual annotations
    sample_df = sample_df[
        [
        "comment_id",
        "comment",
        "detected_language",
        "language_candidate",
        ]
    ]
    sample_df["manual_language"] = ""
    sample_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"Sample audit: {len(sample_df)}")
    print(f"File saved to: {output_path}")
    
if __name__ == "__main__":
    main()