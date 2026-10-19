"""
pilah komenyar yang kemungkinan berbahasa Indonesia termasuk komentar informal yang salah deteksi
"""
from pathlib import Path
import re
import pandas as pd

# project paths
ROOT_DIR = Path(__file__).resolve().parents[2]
input_path = ROOT_DIR / "data" / "interim" / "comments_filterlang.csv"
output_path = ROOT_DIR / "data" / "interim" / "comments_refined.csv"

""" 
indonesian words and informal expressions uased as review clues
not proof that a comment is in Indonesian, but can be used as a clue to refine the language detection
"""
indonesian_clues = {
    "aku", "saya", "kami", "kita", "kalian", "mereka",
    "yang", "dan", "atau", "tapi", "tetapi", "karena",
    "dengan", "untuk", "dari", "ini", "itu", "juga",
    "sudah", "udah", "belum", "akan", "lagi", "masih",
    "enggak", "nggak", "gak", "ga", "aja", "saja",
    "banget", "bgt", "sih", "dong", "deh", "kok",
    "kenapa", "gimana", "bagaimana", "kayak", "kaya",
    "begitu", "bener", "benar", "tolong", "jangan",
    "makasih", "terima", "kasih", "semangat", "keren",
    "sedih", "kasihan", "orang", "percaya", "lihat",
    "menurut", "mungkin", "memang", "padahal", "kalau",
    "kalo", "bang", "mbak", "bu", "pak", "min"
}
def has_indonesian_clues(text):
    """
    check if a comment contains any Indonesian clues
    """
    if pd.isna(text):
        return False
    # normalize text and extract word tokens
    normalized_text = str(text).lower()
    words = set(re.findall(r'\b\w+\b', normalized_text))
    return bool(words & indonesian_clues)

def refine_language(row):
    # assign a review category without deleting any comment
    detected_language = str(row["detected_language".strip().lower()])
    comment = row["comment"]
    
    # keep indonesians' commnt as candidates
    if detected_language == "id":
        return "candidate_id"
    
    # review non-id predicts that contain indonesian clues
    if has_indonesian_clues(comment):
        return "review"
    
    # other predicts remain candidates as for non-id comments
    return "candidate_non_id"

def main():
    # load, refine, and save language detection results
    if not input_path.exists():
        raise FileNotFoundError(
        f"Input file not found: {input_path}"
        "Check the filename and data folder"
        )
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    print("Detected language values: ")
    print(df["detected_language"].value_counts(dropna=False).head(20))
    # check required columns 
    required_columns = {"comment", "detected_language"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )
    
    # add refined categories while preserving all ori columns
    df["language_candidate"] = df.apply(refine_language, axis=1)
    
    # save separate output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    # summary
    print("Refine Language Summary")
    print(f"Input rows: {len(df):,}")
    print("Candidate distribution: ")
    counts = df["language_candidate"].value_counts()
    for category in [
        "candidate_id",
        "review",
        "candidate_non_id"
    ]:
        print(f"{category}: {counts.get(category, 0):,}")
    
    print(f"Output: {output_path}")
    
if __name__ == "__main__":
    main()