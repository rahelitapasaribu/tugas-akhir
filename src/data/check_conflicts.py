
"""
Check duplicate comment IDs and conflicting manual language labels.
Input: four language-audit CSV files in data/interim/
Output: audit_conflicts.csv and audit_unique_comments.csv in data/processed/
"""

from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
INTERIM_DIR = ROOT_DIR / "data" / "interim"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"

AUDIT_FILES = {
    "initial_validation": ("validate_lang.csv", "manual_validation"),
    "review_sample": ("review_sample.csv", "manual_language"),
    "non_id_sample": ("non_id_sample.csv", "manual_language"),
    "id_sample": ("id_sample.csv", "manual_language"),
}

VALID_LABELS = {"id", "non_id", "uncertain"}


def main():
    all_rows = []

    # Read each audit file and standardize the label column.
    for audit_source, (filename, label_column) in AUDIT_FILES.items():
        path = INTERIM_DIR / filename

        if not path.exists():
            print(f"WARNING: File not found: {path}")
            continue

        df = pd.read_csv(path, encoding="utf-8-sig")

        if "comment_id" not in df.columns:
            print(f"WARNING: comment_id missing in {filename}")
            continue

        if label_column not in df.columns:
            print(f"WARNING: {label_column} missing in {filename}")
            continue

        # Keep only necessary fields and standardize values.
        columns = ["comment_id", label_column]
        if "comment" in df.columns:
            columns.append("comment")

        df = df[columns].copy()
        df = df.rename(columns={label_column: "manual_label"})

        df["comment_id"] = df["comment_id"].astype("string").str.strip()
        df["manual_label"] = (
            df["manual_label"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        df["audit_source"] = audit_source

        # Exclude blank or invalid labels from conflict checking.
        df = df[
            df["comment_id"].notna()
            & df["comment_id"].ne("")
            & df["manual_label"].isin(VALID_LABELS)
        ].copy()

        all_rows.append(df)

    if not all_rows:
        raise ValueError("No valid audit data found. Check the input files.")

    combined = pd.concat(all_rows, ignore_index=True)

    # Identify IDs assigned more than one distinct manual label.
    label_counts = combined.groupby("comment_id")["manual_label"].nunique()
    conflict_ids = label_counts[label_counts > 1].index

    conflicts = combined[
        combined["comment_id"].isin(conflict_ids)
    ].sort_values(["comment_id", "audit_source"])

    # Create one record per comment ID for the unique-comment overview.
    # If a conflict exists, mark the final label as needing review.
    unique_rows = []

    for comment_id, group in combined.groupby("comment_id", sort=False):
        labels = sorted(group["manual_label"].unique())
        sources = sorted(group["audit_source"].unique())

        comment_text = ""
        if "comment" in group.columns:
            nonempty = group["comment"].dropna().astype(str)
            if not nonempty.empty:
                comment_text = nonempty.iloc[0]

        unique_rows.append({
            "comment_id": comment_id,
            "comment": comment_text,
            "manual_labels_found": ";".join(labels),
            "audit_sources": ";".join(sources),
            "label_conflict": len(labels) > 1,
            "final_manual_label": labels[0] if len(labels) == 1 else "review",
        })

    unique_comments = pd.DataFrame(unique_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conflict_path = OUTPUT_DIR / "audit_conflicts.csv"
    unique_path = OUTPUT_DIR / "audit_unique_comments.csv"

    conflicts.to_csv(conflict_path, index=False, encoding="utf-8-sig")
    unique_comments.to_csv(unique_path, index=False, encoding="utf-8-sig")

    print("\n=== AUDIT CONFLICT CHECK ===")
    print(f"Total audit rows: {len(combined)}")
    print(f"Unique comment IDs: {combined['comment_id'].nunique()}")
    print(f"Duplicate rows beyond first occurrence: "
          f"{len(combined) - combined['comment_id'].nunique()}")
    print(f"IDs with conflicting labels: {len(conflict_ids)}")
    print(f"Unique comments needing review: "
          f"{unique_comments['label_conflict'].sum()}")

    print("\nUnique label distribution:")
    print(
        unique_comments["final_manual_label"]
        .value_counts()
        .to_string()
    )

    print("\nOutput files:")
    print(f"Conflict details: {conflict_path}")
    print(f"Unique comment overview: {unique_path}")


if __name__ == "__main__":
    main()