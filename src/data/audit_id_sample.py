"""
create random sample to audit fasttext indonesian lang predicts
"""
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT_DIR / "data" / "processed" / "comments_lang_candidates.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "interim" / "id_sample.csv"


def main():
    # Check input file
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

    # Select comments originally classified as Indonesian by FastText
    id_candidates = df[
        df["language_status"] == "candidate_id"
    ].copy()

    # Randomly sample up to 100 comments
    sample_size = min(100, len(id_candidates))
    sample_df = id_candidates.sample(
        n=sample_size,
        random_state=42
    )

    # Prepare blank labels for manual validation
    sample_df = sample_df[
        ["comment_id", "comment", "detected_language"]
    ].copy()
    sample_df["manual_language"] = ""

    # Save sample for annotation
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("Indonesian Candidate Audit Sample")
    print(f"Available candidates: {len(id_candidates)}")
    print(f"Sample size: {len(sample_df)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()