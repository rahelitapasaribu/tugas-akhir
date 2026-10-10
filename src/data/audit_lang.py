"""
tampilin sampel komentar dari hasil deteksi bahasa buat dievaluasi kualitas prediksi pake fasstext
"""
import pandas as pd
input_file = "data/interim/comments_filterlang.csv"
sample_size = 20
random_state = 42

df = pd.read_csv(input_file)
languages = ["id", "en", "ms", "it", "de"]
for language in languages:
    subset = df[df["detected_language"] == language]
    print(f"Language: {language.upper()} - Total comments: {len(subset):,}")
    if subset.empty:
        print("No comments found for this language.")
        continue
    sample = subset.sample(n=min(sample_size, len(subset)), random_state=random_state)
    for i, row in enumerate(sample.itertuples(), start=1):
        print(f"{i}. Comment ID: {row.comment_id}, Confidence: {row.language_confidence:.4f}")
        print(f"   Comment: {row.comment}")
