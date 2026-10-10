"""
mengevaluasi predisksi bahasa fasttext berasarkan validasi manual
"""
from pathlib import Path
import pandas as pd

input_file = Path("data/interim/validate_lang.csv")
output_file = Path("data/interim/evaluate_lang.csv")

def main():
    df = pd.read_csv(input_file)
    
    #validasi label manual 
    valid_label = {"id", "non_id", "uncertain"}
    if df["manual_validation"].isna().any():
        raise ValueError("There are missing values in the 'manual_validation' column.")
    if not set(df["manual_validation"].unique()).issubset(valid_label):
        raise ValueError(f"Invalid values found in 'manual_validation' column.")
    
    # summary of all labels
    print("Language Detection Evaluation")
    print("=" * 70)
    print(f"Total samples: {len(df)}")
    print("Manual validation distribution:")
    print(df["manual_validation"].value_counts().to_string())
    
    # perbanding prediksi fasttext dengan label manual
    comparison = pd.crosstab(df["language_group"], df["manual_validation"])
    comparison = comparison.reindex(columns=["id", "non_id", "uncertain"], fill_value=0)
    comparison["total"] = comparison.sum(axis=1)
    comparison["indonesian_rate"] = (comparison["id"] / comparison["total"] * 100).round(2)
    print("Validation by detected language:")
    print(comparison.to_string())
    
    # evaluasi biner tanpa komentar uncertain
    evaluated = df[df["manual_validation"] != "uncertain"].copy()
    evaluated["predicted_id"] = evaluated["detected_language"] == "id"
    evaluated["actual_id"] = evaluated["manual_validation"] == "id"
    
    tp = ((evaluated["predicted_id"]) & (evaluated["actual_id"])).sum()
    fp = ((evaluated["predicted_id"]) & (~evaluated["actual_id"])).sum()
    fn = ((~evaluated["predicted_id"]) & (evaluated["actual_id"])).sum()
    tn = ((~evaluated["predicted_id"]) & (~evaluated["actual_id"])).sum()
    
    accuracy = (tp + tn) / len(evaluated) if len(evaluated) else 0
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (2 * precision * recall / (precision + recall) if precision + recall else 0)
    
    print("Binary Confusion Matrix:")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives (TN): {tn}")
    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print(
        f"Uncertain excluded: "
        f"{(df['manual_validation'] == 'uncertain').sum()}"
        )
    
    # simpan hasil evaluasi ke file CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_file, index=True, encoding="utf-8-sig")
    print(f"Evaluation results saved to: {output_file}")

if __name__ == "__main__":
    main()
    