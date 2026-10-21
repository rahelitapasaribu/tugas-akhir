"""
evaluasi manual language dari review sample
"""
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "interim"
input_path = DATA_DIR / "review_sample.csv"
detail_path = DATA_DIR / "review_sample_eval.csv"
summary_path = DATA_DIR / "reviewsample_summary.csv"

def main():
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    required = {"comment_id", "detected_language", "manual_language"}
    missing = required - set(df.columns)
    
    if missing: 
        raise ValueError(
            f"Missing column: {sorted(missing)}"
            "Make sure manual_language column is filled"
        )
    
    # normalize annotation labels 
    df["manual_language"] = (
        df["manual_language"].astype(str).str.strip().str.lower()
    )
    df["detected_language"] = (
        df["detected_language"].astype(str).str.strip().str.lower()
    )
    
    valid_label = ["id", "non_id", "uncertain"]
    invalid = ~df["manual_language"].isin(valid_label)
    
    if invalid.any():
        print("Invalid or empty annotations:")
        print(df.loc[invalid, ["comment_id", ["manual_language"]]])
        raise ValueError("Check again manual_language label")
    
    # exclude uncertain labels from binary evaluation
    evaluable = df[df["manual_language"].isin(["id", "non_id"])].copy()
    
    # fasttext predicts inonesian when its detected language is id
    evaluable["fasttext_prediction"] = evaluable["detected_language"].apply(lambda lang: "id" if lang == "id" else "non_id")
    
    # compare predicts with manual annotations
    evaluable["correct"] = (evaluable["fasttext_prediction"].values == evaluable["manual_language"].values)
    
    total = len(evaluable)
    correct = int(evaluable["correct"].sum())
    incorrect = total - correct
    accuracy = correct / total if total else 0
    
    # save detailed results
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    
    #summary
    summary = pd.DataFrame([
        {
            "total_sample": len(df),
            "evaluated_sample": total,
            "uncertain_excluded": int((df["manual_language"] == "unertain").sum()),
            "manual_id": int((df["manual_language"] == "id").sum()),
            "manual_non_id": int((df["manual_language"] == "non_id").sum()),
            "correct_fasttext": correct,
            "incorrect_fasttext": total - correct,
            "accuracy": accuracy,
        }
    ])
    
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("Review Sample Evaluation")
    print("\nManual annotation distribution:")
    print(df["manual_language"].value_counts())

    print(f"\nEvaluated samples: {total}")
    print(f"Correct FastText predictions: {correct}")
    print(f"Incorrect FastText predictions: {total - correct}")
    print(f"Accuracy: {accuracy:.4f}")

    print(f"\nDetailed results: {detail_path}")
    print(f"Summary results: {summary_path}")


if __name__ == "__main__":
    main()