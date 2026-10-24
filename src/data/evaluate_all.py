"""
evaluate all manual lang audit samples
"""
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
INTERIM_DIR = ROOT_DIR / "data" / "interim"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"

AUDIT_FILES = {
    "initial_validation": "validate_lang.csv",
    "review_sample": "review_sample.csv",
    "non_id_sample": "non_id_sample.csv",
    "id_sample": "id_sample.csv",
}

# Map the label column used by each audit file.
LABEL_COLUMNS = {
    "initial_validation": "manual_validation",
    "review_sample": "manual_language",
    "non_id_sample": "manual_language",
    "id_sample": "manual_language",
}

VALID_LABELS = {"id", "non_id", "uncertain"}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_samples = []
    summary_rows = []

    # Read and standardize each audit sample.
    for audit_name, filename in AUDIT_FILES.items():
        file_path = INTERIM_DIR / filename

        if not file_path.exists():
            print(f"WARNING: File not found, skipped: {file_path}")
            continue

        df = pd.read_csv(file_path, encoding="utf-8-sig")

        label_column = LABEL_COLUMNS[audit_name]

        if "comment_id" not in df.columns:
            print(f"WARNING: Missing comment_id in {filename}; skipped.")
            continue

        if label_column not in df.columns:
            print(
                f"WARNING: Missing column '{label_column}' "
                f"in {filename}; skipped."
            )
            continue

        # Standardize labels and preserve the audit source.
        df["manual_label"] = (
            df[label_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df["audit_source"] = audit_name

        # Ignore blank or unexpected labels when calculating metrics.
        valid_df = df[df["manual_label"].isin(VALID_LABELS)].copy()

        counts = valid_df["manual_label"].value_counts()

        id_count = int(counts.get("id", 0))
        non_id_count = int(counts.get("non_id", 0))
        uncertain_count = int(counts.get("uncertain", 0))
        labeled_count = len(valid_df)

        # Calculate ID proportion among determinate labels only.
        determinate_count = id_count + non_id_count
        id_rate = (
            id_count / determinate_count
            if determinate_count > 0
            else float("nan")
        )

        summary_rows.append({
            "audit_source": audit_name,
            "file": filename,
            "rows_in_file": len(df),
            "labeled_rows": labeled_count,
            "id": id_count,
            "non_id": non_id_count,
            "uncertain": uncertain_count,
            "blank_or_invalid": len(df) - labeled_count,
            "id_rate_excluding_uncertain": round(id_rate, 4)
            if pd.notna(id_rate) else None,
        })

        all_samples.append(
            valid_df[["comment_id", "comment", "manual_label", "audit_source"]]
            .copy()
        )

    # Stop if none of the audit files could be evaluated.
    if not all_samples:
        raise ValueError(
            "No audit files could be evaluated. "
            "Check the filenames and label columns."
        )

    # Combine samples and check for overlapping comment IDs.
    combined = pd.concat(all_samples, ignore_index=True)
    combined["comment_id"] = combined["comment_id"].astype(str).str.strip()

    duplicate_mask = combined.duplicated(
        subset=["comment_id"], keep=False
    )
    duplicate_rows = int(duplicate_mask.sum())
    duplicate_ids = combined.loc[
        duplicate_mask, "comment_id"
    ].nunique()

    summary = pd.DataFrame(summary_rows)

    # Save detailed results for further inspection.
    summary_path = OUTPUT_DIR / "audit_summary.csv"
    combined_path = OUTPUT_DIR / "audit_combined.csv"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")

    # Print results.
    print("\n=== LANGUAGE AUDIT SUMMARY BY FILE ===")
    print(summary.to_string(index=False))

    print("\n=== COMBINED AUDIT COUNTS ===")
    print(f"Total labeled rows (including overlaps): {len(combined)}")
    print(f"Unique comment IDs: {combined['comment_id'].nunique()}")
    print(f"Rows belonging to duplicated IDs: {duplicate_rows}")
    print(f"Number of duplicated IDs: {duplicate_ids}")

    print("\nCombined label distribution (rows, not unique comments):")
    print(combined["manual_label"].value_counts().to_string())

    print("\nOutput files:")
    print(f"Summary: {summary_path}")
    print(f"Combined audit: {combined_path}")


if __name__ == "__main__":
    main()
