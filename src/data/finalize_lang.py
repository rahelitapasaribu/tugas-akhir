
"""
Finalize language candidates using automatic predictions and manual audits.
Input: comments_refined.csv and four manual language-audit CSV files.
Output: language candidates, unresolved comments, audited exclusions, and summary CSVs.
"""

from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
INTERIM_DIR = ROOT_DIR / "data" / "interim"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"

REFINED_PATH = INTERIM_DIR / "comments_refined.csv"

AUDIT_FILES = {
    "validate_lang.csv": "manual_validation",
    "review_sample.csv": "manual_language",
    "non_id_sample.csv": "manual_language",
    "id_sample.csv": "manual_language",
}

CANDIDATES_PATH = OUTPUT_DIR / "comments_lang_candidates.csv"
UNRESOLVED_PATH = OUTPUT_DIR / "comments_lang_unresolved.csv"
EXCLUDED_PATH = OUTPUT_DIR / "comments_lang_audited_excluded.csv"
SUMMARY_PATH = OUTPUT_DIR / "finalize_lang_summary.csv"

VALID_LABELS = {"id", "non_id", "uncertain"}


def read_csv_checked(path):
    """Read a CSV file and ensure it exists."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def normalize_id(series):
    """Normalize comment IDs for consistent matching."""
    return series.astype("string").str.strip()


def main():
    # Read and validate the refined comment dataset.
    refined = read_csv_checked(REFINED_PATH)

    required_columns = {
        "comment_id",
        "comment",
        "language_candidate",
    }
    if not required_columns.issubset(refined.columns):
        raise ValueError(
            f"comments_refined.csv must include: {required_columns}"
        )

    refined["comment_id"] = normalize_id(refined["comment_id"])

    if refined["comment_id"].isna().any() or refined["comment_id"].eq("").any():
        raise ValueError("Missing comment_id found in comments_refined.csv.")

    if refined["comment_id"].duplicated().any():
        raise ValueError("Duplicate comment_id found in comments_refined.csv.")

    allowed_groups = {"candidate_id", "review", "candidate_non_id"}
    invalid_groups = set(refined["language_candidate"].dropna()) - allowed_groups

    if invalid_groups or refined["language_candidate"].isna().any():
        raise ValueError(
            f"Unexpected language_candidate values: {invalid_groups}"
        )

    # Load all manual audit labels.
    audit_frames = []

    for filename, label_column in AUDIT_FILES.items():
        audit_df = read_csv_checked(INTERIM_DIR / filename)

        if not {"comment_id", label_column}.issubset(audit_df.columns):
            raise ValueError(
                f"{filename} must include comment_id and {label_column}."
            )

        audit_df = audit_df[["comment_id", label_column]].copy()
        audit_df = audit_df.rename(
            columns={label_column: "manual_language"}
        )
        audit_df["audit_source"] = filename
        audit_df["comment_id"] = normalize_id(audit_df["comment_id"])
        audit_df["manual_language"] = (
            audit_df["manual_language"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        invalid = (
            audit_df["comment_id"].isna()
            | audit_df["comment_id"].eq("")
            | audit_df["manual_language"].isna()
            | ~audit_df["manual_language"].isin(VALID_LABELS)
        )

        if invalid.any():
            print(f"\nInvalid audit rows in {filename}:")
            print(audit_df.loc[invalid].to_string(index=False))
            raise ValueError(f"Fix invalid labels in {filename}.")

        audit_frames.append(audit_df)

    all_audits = pd.concat(audit_frames, ignore_index=True)

    # Ensure the same comment has consistent manual labels across audits.
    label_counts = all_audits.groupby("comment_id")[
        "manual_language"
    ].nunique()

    conflict_ids = label_counts[label_counts > 1].index.tolist()

    if conflict_ids:
        conflict_rows = all_audits[
            all_audits["comment_id"].isin(conflict_ids)
        ].sort_values(["comment_id", "audit_source"])

        print("\nConflicting manual labels detected:")
        print(conflict_rows.to_string(index=False))
        raise ValueError(
            "Resolve all manual-label conflicts before finalization."
        )

    # One consistent manual label per comment ID.
    manual_labels = (
        all_audits.drop_duplicates(subset="comment_id")
        [["comment_id", "manual_language"]]
        .copy()
    )

    # Attach manual labels to the refined dataset.
    data = refined.merge(
        manual_labels,
        on="comment_id",
        how="left",
        validate="one_to_one",
    )

    # Manual labels take priority over automatic language predictions.
    manually_confirmed_id = data["manual_language"].eq("id")
    manually_excluded = data["manual_language"].isin(
        ["non_id", "uncertain"]
    )
    unreviewed = data["manual_language"].isna()

    # Comments classified as Indonesian by manual audit.
    manual_id = data[manually_confirmed_id].copy()
    manual_id["language_status"] = "manual_id_in_audit_sample"

    # Automatic Indonesian candidates not manually audited.
    automatic_id = data[
        unreviewed
        & data["language_candidate"].eq("candidate_id")
    ].copy()
    automatic_id["language_status"] = "candidate_id"

    # Review-group comments not manually audited remain provisional.
    review_candidates = data[
        unreviewed
        & data["language_candidate"].eq("review")
    ].copy()
    review_candidates["language_status"] = "review_not_fully_verified"

    # Unreviewed non-Indonesian predictions remain unresolved.
    unresolved = data[
        unreviewed
        & data["language_candidate"].eq("candidate_non_id")
    ].copy()
    unresolved["language_status"] = "unreviewed_candidate_non_id"

    # Manually labeled non-Indonesian or uncertain comments are excluded.
    excluded = data[manually_excluded].copy()
    excluded["language_status"] = "manual_non_id_or_uncertain"

    # Combine all candidate groups.
    candidates = pd.concat(
        [manual_id, automatic_id, review_candidates],
        ignore_index=True,
    )

    # Verify that every refined comment appears in exactly one output.
    candidate_ids = set(candidates["comment_id"].dropna())
    unresolved_ids = set(unresolved["comment_id"].dropna())
    excluded_ids = set(excluded["comment_id"].dropna())

    if candidate_ids & unresolved_ids:
        raise ValueError("Overlap found between candidates and unresolved.")
    if candidate_ids & excluded_ids:
        raise ValueError("Overlap found between candidates and excluded.")
    if unresolved_ids & excluded_ids:
        raise ValueError("Overlap found between unresolved and excluded.")

    output_ids = candidate_ids | unresolved_ids | excluded_ids
    refined_ids = set(refined["comment_id"].dropna())

    missing_ids = refined_ids - output_ids
    unexpected_ids = output_ids - refined_ids

    if missing_ids or unexpected_ids:
        raise ValueError(
            f"Coverage check failed: {len(missing_ids)} missing IDs, "
            f"{len(unexpected_ids)} unexpected IDs."
        )

    # Build a summary using actual output counts.
    summary_rows = [
        {
            "category": "total_refined_comments",
            "count": len(refined),
        },
        {
            "category": "manually_confirmed_id",
            "count": len(manual_id),
        },
        {
            "category": "automatic_id_not_manually_audited",
            "count": len(automatic_id),
        },
        {
            "category": "review_not_fully_verified",
            "count": len(review_candidates),
        },
        {
            "category": "total_language_candidates",
            "count": len(candidates),
        },
        {
            "category": "unreviewed_candidate_non_id",
            "count": len(unresolved),
        },
        {
            "category": "manual_non_id_or_uncertain",
            "count": len(excluded),
        },
        {
            "category": "total_output_comments",
            "count": len(output_ids),
        },
    ]

    summary = pd.DataFrame(summary_rows)

    # Save all outputs.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates.to_csv(
        CANDIDATES_PATH, index=False, encoding="utf-8-sig"
    )
    unresolved.to_csv(
        UNRESOLVED_PATH, index=False, encoding="utf-8-sig"
    )
    excluded.to_csv(
        EXCLUDED_PATH, index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        SUMMARY_PATH, index=False, encoding="utf-8-sig"
    )

    # Print final counts and verification results.
    print("\n=== LANGUAGE FINALIZATION SUMMARY ===")
    print(summary.to_string(index=False))

    print("\n=== CANDIDATE STATUS DISTRIBUTION ===")
    print(candidates["language_status"].value_counts().to_string())

    print("\n=== COVERAGE CHECK ===")
    print(f"Refined rows: {len(refined)}")
    print(f"Candidate rows: {len(candidates)}")
    print(f"Unresolved rows: {len(unresolved)}")
    print(f"Excluded rows: {len(excluded)}")
    print(f"Unique output IDs: {len(output_ids)}")
    print(f"Missing IDs: {len(missing_ids)}")
    print(f"Unexpected IDs: {len(unexpected_ids)}")
    print("PASS: Each refined comment belongs to exactly one output group.")

    print("\nOutput files:")
    print(f"Candidates: {CANDIDATES_PATH}")
    print(f"Unresolved: {UNRESOLVED_PATH}")
    print(f"Excluded: {EXCLUDED_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()