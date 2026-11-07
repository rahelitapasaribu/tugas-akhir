
# Mengambil 10.000 komentar dengan distribusi label dan pembagian dataset sesuai arahan dosen.

import pandas as pd
from pathlib import Path

INPUT_FILE = "data/processed/comments_labeled.csv"
OUTPUT_DIR = Path("data/processed")
SPLIT_DIR = OUTPUT_DIR / "splits_custom"

LABEL_COLUMN = "sentiment_qwen"
RANDOM_STATE = 42

# Target jumlah data setiap label pada masing-masing subset
SPLIT_TARGETS = {
    "Netral": {
        "train": 2427,
        "validation": 515,
        "test": 525
    },
    "Negatif": {
        "train": 2341,
        "validation": 500,
        "test": 505
    },
    "Positif": {
        "train": 2232,
        "validation": 485,
        "test": 470
    }
}

print("Membaca dataset...")

df = pd.read_csv(
    INPUT_FILE,
    dtype={"comment_id": str},
    low_memory=False
)

# Memeriksa kelengkapan dan keunikan ID
if df["comment_id"].isna().any():
    raise ValueError("Terdapat comment_id kosong.")

if df["comment_id"].duplicated().any():
    raise ValueError("Terdapat comment_id duplikat.")

# Memeriksa jumlah data setiap label
print("\nMemeriksa ketersediaan data...")

for label, targets in SPLIT_TARGETS.items():
    required = sum(targets.values())
    available = int((df[LABEL_COLUMN] == label).sum())

    print(f"{label}: tersedia {available}, dibutuhkan {required}")

    if available < required:
        raise ValueError(f"Data {label} tidak mencukupi.")

# Menyiapkan tempat penyimpanan
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

train_parts = []
validation_parts = []
test_parts = []
sampled_parts = []

# Sampling secara acak berdasarkan target setiap label
for label, targets in SPLIT_TARGETS.items():

    total_target = sum(targets.values())

    class_sample = df[
        df[LABEL_COLUMN] == label
    ].sample(
        n=total_target,
        random_state=RANDOM_STATE
    ).copy()

    sampled_parts.append(class_sample)

    # Mengambil data training
    train_data = class_sample.sample(
        n=targets["train"],
        random_state=RANDOM_STATE
    )

    remaining = class_sample.drop(train_data.index)

    # Mengambil data validation
    validation_data = remaining.sample(
        n=targets["validation"],
        random_state=RANDOM_STATE
    )

    # Sisa data digunakan untuk testing
    test_data = remaining.drop(validation_data.index)

    train_parts.append(train_data)
    validation_parts.append(validation_data)
    test_parts.append(test_data)

# Menggabungkan setiap subset dan mengacak urutannya
train_df = pd.concat(train_parts).sample(
    frac=1, random_state=RANDOM_STATE
).reset_index(drop=True)

validation_df = pd.concat(validation_parts).sample(
    frac=1, random_state=RANDOM_STATE
).reset_index(drop=True)

test_df = pd.concat(test_parts).sample(
    frac=1, random_state=RANDOM_STATE
).reset_index(drop=True)

sampled_df = pd.concat(sampled_parts).sample(
    frac=1, random_state=RANDOM_STATE
).reset_index(drop=True)

# Menyimpan hasil sampling dan pembagian
sampled_df.to_csv(
    OUTPUT_DIR / "comments_custom_10000.csv",
    index=False,
    encoding="utf-8-sig"
)

train_df.to_csv(
    SPLIT_DIR / "train.csv",
    index=False,
    encoding="utf-8-sig"
)

validation_df.to_csv(
    SPLIT_DIR / "validation.csv",
    index=False,
    encoding="utf-8-sig"
)

test_df.to_csv(
    SPLIT_DIR / "test.csv",
    index=False,
    encoding="utf-8-sig"
)

# Memeriksa distribusi aktual
print("\n========== HASIL PEMBAGIAN DATA ==========")

for name, data in [
    ("TRAINING", train_df),
    ("VALIDATION", validation_df),
    ("TESTING", test_df)
]:
    print(f"\n{name}: {len(data)} data")
    print(data[LABEL_COLUMN].value_counts().reindex(
        ["Netral", "Negatif", "Positif"], fill_value=0
    ))

# Memastikan tidak ada ID yang tumpang tindih
train_ids = set(train_df["comment_id"])
validation_ids = set(validation_df["comment_id"])
test_ids = set(test_df["comment_id"])

assert train_ids.isdisjoint(validation_ids)
assert train_ids.isdisjoint(test_ids)
assert validation_ids.isdisjoint(test_ids)

assert len(train_df) == 7000
assert len(validation_df) == 1500
assert len(test_df) == 1500
assert len(sampled_df) == 10000

print("\nPemeriksaan ID antar-split: LULUS")
print("Pemeriksaan jumlah data: LULUS")
print("Seluruh file berhasil disimpan.")
print("==========================================")
