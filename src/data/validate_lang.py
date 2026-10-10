"""
Menyiapkan sampel validasi manual untuk mengevaluasi hasil
deteksi bahasa fastText pada komentar YouTube.
"""
from pathlib import Path
import pandas as pd

input_file = Path("data/interim/comments_filterlang.csv")
output_file = Path("data/interim/validate_lang.csv")

RANDOM_STATE = 42
sample_sizes = {
    "id": 50,
    "en": 50,
    "ms": 50,
    "de": 50,
    "it": 50,
    "other": 100
}

def main():
    df = pd.read_csv(input_file, dtype={"comment_id": str})
    
    # kelompokkan komentar berdasarkan bahasa
    df["language_group"] = df["detected_language"].where(df["detected_language"].isin(["id", "en", "ms", "de", "it"]), "other")
    samples = []
    
    # ambil sampel 
    for group, size in sample_sizes.items():
        subset = df[df["language_group"] == group]
        if subset.empty:
            continue
        sample = subset.sample(n=min(size, len(subset)), random_state=RANDOM_STATE)
        samples.append(sample)
    
    review = pd.concat(samples, ignore_index=True)
    
    # siapkan kolom untuk validasi manual
    review["manual_validation"] = ""
    review["validation_notes"] = ""
    
    column_order = ["comment_id", "comment", "detected_language", "language_group", "manual_validation", "validation_notes"]
    review = review[column_order]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(output_file, index=False)
    
    print("Language validation sample")
    print(f"Total samples: {len(review)}")
    print("Samples per language group:")
    print(review["language_group"].value_counts().to_string())
    print(f"Sample saved to: {output_file}")

if __name__ == "__main__":
    main()