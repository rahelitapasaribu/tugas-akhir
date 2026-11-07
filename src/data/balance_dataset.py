
# Mengambil 10.000 komentar secara seimbang dan membaginya menjadi train, validation, dan test.

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Lokasi file
INPUT_FILE = "data/processed/comments_labeled.csv"
OUTPUT_DIR = Path("data/processed")

# Pengaturan sampling
TARGET_COUNTS = {
    "Positif": 3334,
    "Negatif": 3333,
    "Netral": 3333
}

RANDOM_STATE = 42
LABEL_COLUMN = "sentiment_qwen"

print("Membaca dataset...")
df = pd.read_csv(
    INPUT_FILE,
    dtype={"comment_id": str},
    low_memory=False
)

# Memastikan label valid
valid_labels = set(TARGET_COUNTS.keys())
actual_labels = set(df[LABEL_COLUMN].dropna().unique())

if actual_labels != valid_labels:
    raise ValueError(
        f"Label dataset tidak sesuai. Ditemukan: {actual_labels}"
    )

# Memastikan jumlah setiap kelas mencukupi
print("\nMemeriksa jumlah data setiap kelas...")

for label, target in TARGET_COUNTS.items():
    available = int((df[LABEL_COLUMN] == label).sum())
    print(f"{label}: tersedia {available}, target {target}")

    if available < target:
        raise ValueError(
            f"Data {label} tidak mencukupi untuk sampling."
        )

# Sampling acak sesuai target setiap kelas
sampled_parts = []

for label, target in TARGET_COUNTS.items():
    class_data = df[df[LABEL_COLUMN] == label]

    class_sample = class_data.sample(
        n=target,
        random_state=RANDOM_STATE
    )

    sampled_parts.append(class_sample)

# Menggabungkan dan mengacak 10.000 data
sampled_df = pd.concat(
    sampled_parts,
    ignore_index=True
)

sampled_df = sampled_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

print("\nTotal hasil sampling:", len(sampled_df))

# Membagi data: 70% train, 15% validation, 15% test
train_val_df, test_df = train_test_split(
    sampled_df,
    test_size=1500,
    stratify=sampled_df[LABEL_COLUMN],
    random_state=RANDOM_STATE
)

train_df, val_df = train_test_split(
    train_val_df,
    test_size=1500,
    stratify=train_val_df[LABEL_COLUMN],
    random_state=RANDOM_STATE
)

# Membuat folder penyimpanan
split_dir = OUTPUT_DIR / "splits"
split_dir.mkdir(parents=True, exist_ok=True)

# Menyimpan dataset hasil sampling dan pembagian
sampled_df.to_csv(
    OUTPUT_DIR / "comments_balanced_10000.csv",
    index=False,
    encoding="utf-8-sig"
)

train_df.to_csv(
    split_dir / "train.csv",
    index=False,
    encoding="utf-8-sig"
)

val_df.to_csv(
    split_dir / "validation.csv",
    index=False,
    encoding="utf-8-sig"
)

test_df.to_csv(
    split_dir / "test.csv",
    index=False,
    encoding="utf-8-sig"
)

# Menampilkan hasil pembagian
print("\n========== HASIL PEMBAGIAN DATA ==========")
print(f"Training   : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Testing    : {len(test_df)}")
print(f"Total      : {len(train_df) + len(val_df) + len(test_df)}")

print("\n========== DISTRIBUSI LABEL ==========")

for name, data in [
    ("TRAINING", train_df),
    ("VALIDATION", val_df),
    ("TESTING", test_df)
]:
    print(f"\n{name}")
    print(data[LABEL_COLUMN].value_counts().sort_index())

# Memastikan tidak ada ID yang tumpang tindih
train_ids = set(train_df["comment_id"])
val_ids = set(val_df["comment_id"])
test_ids = set(test_df["comment_id"])

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

print("\nPemeriksaan ID antar-split: LULUS")
print("Seluruh file berhasil disimpan.")
print("==========================================")
