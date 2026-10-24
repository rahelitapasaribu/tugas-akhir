
"""
Merge refined language candidates with manual language annotations.
Input: comments_refined.csv, review_sample.csv, non_id_sample.csv
Output: comments_lang_candidates.csv, comments_lang_unresolved.csv, finalize_lang_summary.csv
"""
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "interim"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"

REFINED_PATH = DATA_DIR / "comments_refined.csv"
REVIEW_PATH = DATA_DIR / "review_sample.csv"
NON_ID_PATH = DATA_DIR / "non_id_sample.csv"

CANDIDATES_PATH = OUTPUT_DIR / "comments_lang_candidates.csv"
UNRESOLVED_PATH = OUTPUT_DIR / "comments_lang_unresolved.csv"
SUMMARY_PATH = OUTPUT_DIR / "finalize_lang_summary.csv"

VALID_LABELS = {"id", "non_id", "uncertain"}


def read_csv_checked(path):
    """Read a CSV file and check that it exists."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def normalize_id(series):
    """Normalize comment IDs for consistent matching."""
    return series.astype("string").str.strip()


def main():
    # Read input files
    refined = read_csv_checked(REFINED_PATH)
    review = read_csv_checked(REVIEW_PATH)
    non_id = read_csv_checked(NON_ID_PATH)

    # Validate required columns
    required_refined = {"comment_id", "language_candidate", "comment"}
    required_audit = {"comment_id", "manual_language"}

    if not required_refined.issubset(refined.columns):
        raise ValueError(
            f"comments_refined.csv must include: {required_refined}"
        )

    for name, df in [
        ("review_sample.csv", review),
        ("non_id_sample.csv", non_id),
    ]:
        if not required_audit.issubset(df.columns):
            raise ValueError(
                f"{name} must include columns: {required_audit}"
            )

    # Normalize IDs and manual labels
    refined["comment_id"] = normalize_id(refined["comment_id"])

    for name, df in [
        ("review_sample.csv", review),
        ("non_id_sample.csv", non_id),
    ]:
        df["comment_id"] = normalize_id(df["comment_id"])
        df["manual_language"] = (
            df["manual_language"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        invalid = (
            df["manual_language"].isna()
            | ~df["manual_language"].isin(VALID_LABELS)
        )

        if invalid.any():
            print(f"\nInvalid labels in {name}:")
            print(
                df.loc[
                    invalid,
                    ["comment_id", "manual_language"]
                ].to_string(index=False)
            )
            raise ValueError(
                f"Fix invalid manual_language labels in {name}."
            )

        if df["comment_id"].duplicated().any():
            raise ValueError(
                f"Duplicate comment_id found in {name}."
            )

    # Prepare initial language candidate groups
    candidate_id = refined[
        refined["language_candidate"] == "candidate_id"
    ].copy()
    candidate_id["language_status"] = "candidate_id"

    review_candidates = refined[
        refined["language_candidate"] == "review"
    ].copy()
    review_candidates["language_status"] = "review_not_fully_verified"

    # Add only manually confirmed Indonesian comments from both audits
    review_id_ids = set(
        review.loc[
            review["manual_language"] == "id", "comment_id"
        ].dropna()
    )

    non_id_sample_id_ids = set(
        non_id.loc[
            non_id["manual_language"] == "id", "comment_id"
        ].dropna()
    )

    audited_id_ids = review_id_ids | non_id_sample_id_ids

    audited_id_candidates = refined[
        refined["comment_id"].isin(audited_id_ids)
    ].copy()
    audited_id_candidates["language_status"] = "manual_id_in_audit_sample"

    # Combine candidate groups and deduplicate by comment ID
    candidates = pd.concat(
        [candidate_id, review_candidates, audited_id_candidates],
        ignore_index=True
    )
    candidates = candidates.drop_duplicates(
        subset="comment_id", keep="first"
    )

    # Keep candidate_non_id comments without manual audit labels unresolved
    audited_ids = (
        set(review["comment_id"].dropna())
        | set(non_id["comment_id"].dropna())
    )

    unresolved = refined[
        (refined["language_candidate"] == "candidate_non_id")
        & (~refined["comment_id"].isin(audited_ids))
    ].copy()
    unresolved["language_status"] = "unreviewed_candidate_non_id"

    # Create a summary of the processing results
    summary = pd.DataFrame([
        {
            "category": "candidate_id_initial",
            "count": len(candidate_id),
        },
        {
            "category": "review_candidates",
            "count": len(review_candidates),
        },
        {
            "category": "manually_confirmed_id_from_audits",
            "count": len(audited_id_candidates),
        },
        {
            "category": "unreviewed_candidate_non_id",
            "count": len(unresolved),
        },
        {
            "category": "total_language_candidates",
            "count": len(candidates),
        },
    ])

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates.to_csv(
        CANDIDATES_PATH, index=False, encoding="utf-8-sig"
    )
    unresolved.to_csv(
        UNRESOLVED_PATH, index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        SUMMARY_PATH, index=False, encoding="utf-8-sig"
    )

    print("\nLanguage Finalization Summary")
    print(summary.to_string(index=False))
    print(f"\nCandidates: {CANDIDATES_PATH}")
    print(f"Unresolved: {UNRESOLVED_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()