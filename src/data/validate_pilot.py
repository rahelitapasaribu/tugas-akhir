# Memvalidasi label Ollama terhadap label manual, menghitung accuracy, confusion matrix, classification report, distribusi sentimen, dan menyimpan komentar dengan label berbeda.

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("data/processed/sentiment_pilot.csv")

for col in ["sentiment_llm", "manual_sentiment"]:
    df[col] = df[col].astype("string").str.lower().str.strip()

labels = ["positif", "negatif", "netral"]

valid = df[
    df["sentiment_llm"].isin(labels) &
    df["manual_sentiment"].isin(labels)
].copy()

print("\n===== HASIL VALIDASI LABEL PILOT =====")
print("Jumlah seluruh komentar:", len(df))
print("Jumlah komentar dengan kedua label valid:", len(valid))
print("Jumlah label kosong/tidak valid:", len(df) - len(valid))

if len(valid) > 0:
    accuracy = accuracy_score(valid["manual_sentiment"], valid["sentiment_llm"])
    print("\nAccuracy:", round(accuracy, 4))

    print("\nClassification Report:")
    print(classification_report(valid["manual_sentiment"], valid["sentiment_llm"], labels=labels, zero_division=0))

    print("\nConfusion Matrix:")
    print(pd.DataFrame(confusion_matrix(valid["manual_sentiment"], valid["sentiment_llm"], labels=labels), index=[f"Manual {x}" for x in labels], columns=[f"Ollama {x}" for x in labels]))

    mismatch = valid[valid["manual_sentiment"] != valid["sentiment_llm"]]
    mismatch.to_csv("data/processed/pilot_label_mismatch.csv", index=False)
    print("\nJumlah label berbeda:", len(mismatch))
    print("Komentar yang berbeda disimpan di: data/processed/pilot_label_mismatch.csv")

print("\n===== DISTRIBUSI LABEL MANUAL =====")
print(df["manual_sentiment"].value_counts(dropna=False))
print("\nPersentase label manual:")
print((df["manual_sentiment"].value_counts(normalize=True, dropna=False) * 100).round(2))