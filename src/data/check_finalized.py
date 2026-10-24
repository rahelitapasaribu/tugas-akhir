"""
check finalized language candidate files for duplicates and status consistency
"""
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"
candidates_path = DATA_DIR / "comments_lang_candidates.csv"
unresolved_path = DATA_DIR / "comments_lang_unresolved.csv"

def main():
    candidates = pd.read_csv(candidates_path, encoding="utf-8-sig")
    unresolved = pd.read_csv(unresolved_path, encoding="utf-8-sig")
    print("Finalized Language Data Check")
    print(f"Candidate rows: {len(candidates)}")
    print(f"Unique candidate ids: {candidates["comment_id"].nunique()}")
    print(f"Unresolved rows: {len(unresolved)}")
    print(f"Unique unresolved ids: {unresolved["comment_id"].nunique()}")
    print("\nCandidate status distribution: ")
    print(candidates["language_status"].value_counts(dropna=False).to_string())
    
    # verify ids dont overlap between the 2 outputs
    candidate_ids = set(candidates["comment_id"].astype(str))
    unresolved_ids = set(unresolved["comment_id"].astype(str))
    overlap = candidate_ids & unresolved_ids
    
    print(f"\nDuplicate candidate ids: {candidates["comment_id"].duplicated().sum()}")
    print(f"Overlap betwwen candidates & unresolved: {len(overlap)}")
    
    if candidates["comment_id"].duplicated().any():
        print("Duplicate ids exist in candidates")
    if unresolved["comment_id"].duplicated().any():
        print("Duplicate ids exist in unresolved")
    if overlap:
        print("Some ids appear in both output files")
    if (
        not candidates["comment_id"].duplicated().any()
        and not unresolved["comment_id"].duplicated().any()
        and not overlap
    ):
        print("No duplicate ids or overlap between output files")
        
if __name__ == "__main__":
    main()