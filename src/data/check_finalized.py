
"""
Check language candidate outputs for duplicate IDs and complete coverage.
Input: comments_refined.csv and three finalized language output files
Output: validation report in the terminal
"""
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
INTERIM_DIR = ROOT_DIR / "data" / "interim"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

REFINED_PATH = INTERIM_DIR / "comments_refined.csv"
CANDIDATES_PATH = PROCESSED_DIR / "comments_lang_candidates.csv"
UNRESOLVED_PATH = PROCESSED_DIR / "comments_lang_unresolved.csv"
EXCLUDED_PATH = PROCESSED_DIR / "comments_lang_audited_excluded.csv"


def load_csv(path):
    """Read a CSV file and report a clear error if missing."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def normalize_ids(df):
    """Normalize comment IDs for reliable comparison."""
    return set(df["comment_id"].astype("string").str.strip().dropna())


def main():
    # Load source and finalized output files
    refined = load_csv(REFINED_PATH)
    candidates = load_csv(CANDIDATES_PATH)
    unresolved = load_csv(UNRESOLVED_PATH)
    excluded = load_csv(EXCLUDED_PATH)

    # Check row counts and unique IDs
    print("Finalized Language Data Check")
    print(f"Refined rows: {len(refined)}")
    print(f"Candidate rows: {len(candidates)}")
    print(f"Unique candidate IDs: {candidates['comment_id'].nunique()}")
    print(f"Unresolved rows: {len(unresolved)}")
    print(f"Unique unresolved IDs: {unresolved['comment_id'].nunique()}")
    print(f"Excluded rows: {len(excluded)}")
    print(f"Unique excluded IDs: {excluded['comment_id'].nunique()}")

    # Show candidate status distribution
    print("\nCandidate status distribution:")
    print(
        candidates["language_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    # Check duplicates within each output
    print("\nDuplicate ID counts:")
    for name, df in [
        ("candidates", candidates),
        ("unresolved", unresolved),
        ("excluded", excluded),
    ]:
        duplicates = df["comment_id"].duplicated().sum()
        print(f"{name}: {duplicates}")

    # Check overlaps between output files
    candidate_ids = normalize_ids(candidates)
    unresolved_ids = normalize_ids(unresolved)
    excluded_ids = normalize_ids(excluded)

    overlap_candidate_unresolved = candidate_ids & unresolved_ids
    overlap_candidate_excluded = candidate_ids & excluded_ids
    overlap_unresolved_excluded = unresolved_ids & excluded_ids

    print("\nOverlap checks:")
    print(f"Candidates vs unresolved: {len(overlap_candidate_unresolved)}")
    print(f"Candidates vs excluded: {len(overlap_candidate_excluded)}")
    print(f"Unresolved vs excluded: {len(overlap_unresolved_excluded)}")

    # Display IDs that appear in both candidates and excluded
    overlap_ids = candidate_ids & excluded_ids

    if overlap_ids:
        print("\nIDs overlapping between candidates and excluded:")
        print(
            refined[
                refined["comment_id"].astype("string").str.strip().isin(overlap_ids)
            ][["comment_id", "language_candidate", "detected_language", "comment"]]
            .to_string(index=False)
    )
    # Check whether every refined comment appears in an output
    refined_ids = normalize_ids(refined)
    all_output_ids = candidate_ids | unresolved_ids | excluded_ids
    missing_ids = refined_ids - all_output_ids
    extra_ids = all_output_ids - refined_ids

    print("\nFull coverage check:")
    print(f"Unique refined IDs: {len(refined_ids)}")
    print(f"IDs missing from all outputs: {len(missing_ids)}")
    print(f"IDs not found in refined data: {len(extra_ids)}")

    if missing_ids:
        missing_rows = refined[
            refined["comment_id"].astype("string").str.strip().isin(missing_ids)
        ]
        print("\nMissing comment details:")
        print(
            missing_rows[
                ["comment_id", "language_candidate", "detected_language"]
            ].to_string(index=False)
        )

    # Print final validation result
    no_duplicates = all(
        not df["comment_id"].duplicated().any()
        for df in [candidates, unresolved, excluded]
    )
    no_overlap = not (
        overlap_candidate_unresolved
        or overlap_candidate_excluded
        or overlap_unresolved_excluded
    )
    complete_coverage = not missing_ids and not extra_ids

    if no_duplicates and no_overlap and complete_coverage:
        print("\nPASS: No duplicates, no overlap, and full coverage.")
    else:
        print("\nCHECK NEEDED: Review the issues reported above.")


if __name__ == "__main__":
    main()
