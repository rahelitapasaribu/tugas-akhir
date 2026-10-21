"""
evaluasi hasil refinement pakai 350 komentar yang sudah di anotasi manual 
"""
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "interim"
refined_path = DATA_DIR / "comments_refined.csv"
val_path = DATA_DIR / "validate_lang.csv"
output_path = DATA_DIR /"refinement_evaluation.csv"
summary_path = DATA_DIR / "refinement_summary.csv"

def calculate_metrics(actual, predicted):
    tp = ((actual == 1) & (predicted == 1)).sum()
    fp = ((actual == 0) & (predicted == 1)).sum()
    fn = ((actual == 1) & (predicted == 0)).sum()
    tn = ((actual == 0) & (predicted == 0)).sum()
    
    accuracy = (tp + tn) / len(actual) if len(actual) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) else 0)
    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def main():
    # compare fasttext and refinement against manual annotations
    for path in [refined_path, val_path]:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
    
    # load dataset
    refined = pd.read_csv(refined_path, encoding="utf-8-sig")
    val = pd.read_csv(val_path, encoding="utf-8-sig")

    # validate required columns
    required_refined = {
        "comment_id",
        "detected_language",
        "language_candidate",
    }
    required_val = {"comment_id", "manual_validation"}
    missing_refined = required_refined - set(refined.columns)
    missing_val = required_val - set(val.columns)
    if missing_refined:
        raise ValueError(f"Missing columns in refined dataset: {sorted(missing_refined)}")
    if missing_val:
        raise ValueError(f"Missing columns in validation dataset: " f"{sorted(missing_val)}")
    
    # normalize ids and manual label before merge
    refined["comment_id"] = refined["comment_id"].astype(str).str.strip()
    val["comment_id"] = (val["comment_id"].astype(str).str.strip())
    val["manual_validation"] = (val["manual_validation"].astype(str).str.strip().str.lower())
    val = val.drop_duplicates(subset="comment_id")
    merged = refined.merge(
        val[["comment_id", "manual_validation"]],
        on="comment_id",
        how="inner",
        validate="one_to_one",
    )
    
    if merged.empty:
        raise ValueError("No matching comment IDs. Check whether both file use" "the same comment_id values")
    
    evaluable = merged[merged["manual_validation"].isin(["id", "non_id"])].copy()
    
    if evaluable.empty:
        raise ValueError("No id/non_id annotations available for evaluation")
    
    actual = (evaluable["manual_validation"] == "id").astype(int)
    
    fasttext_pred = (evaluable["detected_language"].astype(str).str.strip().str.lower() == "id").astype(int)
    
    refined_pred = evaluable["language_candidate"].isin(
        ["candidate_id", "review"]
    ).astype(int)
    
    # calculate metrics
    fasttext_metric =calculate_metrics(actual, fasttext_pred)
    refined_metric = calculate_metrics(actual, refined_pred)
    summary_rows = []
    
    for method, metrics in [
        ("FastText ID only", fasttext_metric),
        ("Refinement ID plus review", refined_metric),
    ]:
        summary_rows.append({
            "method": method,
            "evaluated_samples": len(evaluable),
            "uncertain_excluded": int(
                (merged["manual_validation"] == "uncertain").sum()
            ),
            **metrics,
        })
    
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig",)
    
    # Show results
    print("Language Refinement Evaluation")
    print(f"Matched annotations: {len(merged)}")
    print(f"Evaluated samples: {len(evaluable)}")
    print(
        "Uncertain annotations excluded: "
        f"{(merged['manual_validation'] == 'uncertain').sum()}"
    )

    print("\nManual label distribution:")
    print(merged["manual_validation"].value_counts())

    print("\nMetrics comparison:")
    print(
        summary[
            [
                "method",
                "TP",
                "FP",
                "FN",
                "TN",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print(f"\nDetailed results saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
