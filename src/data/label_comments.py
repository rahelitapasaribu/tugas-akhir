
# Melabeli seluruh komentar berbahasa Indonesia menggunakan Qwen 2.5:3b melalui Ollama, menghasilkan label dan alasan, serta menyimpan progres secara berkala.

import json
import time
from pathlib import Path

import pandas as pd
import ollama

# 1. KONFIGURASI
INPUT_FILE = Path("data/processed/comments_lang_candidates.csv")
OUTPUT_FILE = Path("data/processed/comments_labeled.csv")

MODEL_NAME = "qwen2.5:3b"
SAVE_EVERY = 50
MAX_RETRIES = 3

VALID_LABELS = {"Positif", "Negatif", "Netral"}

# 2. PROMPT ANOTASI
SYSTEM_PROMPT = """
Kamu adalah anotator sentimen untuk penelitian analisis sentimen
komentar YouTube mengenai kasus Kim Sae Ron dan Kim Soo Hyun.

Tugasmu adalah menentukan satu label sentimen untuk setiap komentar
berdasarkan isi komentar dan konteks pembahasan yang tersedia.

KATEGORI SENTIMEN:

1. Positif:
Komentar yang menunjukkan dukungan, apresiasi, simpati,
antusiasme, atau penilaian positif terhadap objek pembahasan.

2. Negatif:
Komentar yang menunjukkan kritik, kemarahan, kekecewaan,
tuduhan, penolakan, atau sindiran negatif terhadap
objek pembahasan.

3. Netral:
Komentar berupa pertanyaan, informasi, pernyataan faktual,
rasa penasaran, atau tidak menunjukkan kecenderungan
positif maupun negatif yang jelas.

ATURAN ANOTASI:

1. Setiap komentar hanya memiliki satu label dominan.
2. Pertimbangkan konteks dan maksud komentar.
3. Emoji tidak boleh menjadi satu-satunya dasar penentuan label.
4. Pertanyaan tidak otomatis dikategorikan negatif.
5. Sarkasme dinilai berdasarkan maksud kalimat.
6. Komentar tanpa kecenderungan sentimen yang jelas
   dikategorikan sebagai Netral.
7. Jika terdapat beberapa sentimen, pilih sentimen dominan.
8. Penentuan sentimen didasarkan pada sikap atau evaluasi
   terhadap objek pembahasan, bukan hanya kata positif
   atau negatif yang muncul.
9. Ekspresi emosi tanpa kecenderungan evaluasi yang jelas
   terhadap objek pembahasan dikategorikan Netral.
10. Pertanyaan retoris yang mengandung kritik, tuduhan,
    atau sindiran negatif dikategorikan Negatif.
11. Komentar dengan sentimen campuran diberi label berdasarkan
    sentimen yang paling dominan.
12. Sarkasme atau pujian yang digunakan untuk menyampaikan
    ejekan dan kritik dikategorikan Negatif.
13. Komentar spam, promosi, dan kurang relevan tetap diberi label.
14. Kata positif atau negatif dalam promosi tidak otomatis
    menentukan label sentimen.
15. Jika maksud komentar tidak dapat ditentukan,
    kategorikan sebagai Netral.
16. Nilai setiap komentar secara mandiri.
17. Jangan mengarang konteks yang tidak terdapat dalam komentar.

FORMAT JAWABAN:

Berikan jawaban hanya dalam format JSON berikut:

{
    "sentiment": "Positif",
    "reason": "Alasan singkat berdasarkan isi komentar."
}

KETENTUAN OUTPUT:

- Label hanya boleh Positif, Negatif, atau Netral.
- Alasan harus singkat, spesifik, dan sesuai isi komentar.
- Jangan memberikan penjelasan di luar JSON.
- Jangan menambahkan kategori sentimen lain.
"""

# 3. FUNGSI MEMBACA RESPONS QWEN
def parse_response(response):
    try:
        content = response.strip()

        # Mengantisipasi JSON yang dibungkus markdown
        if content.startswith("```"):
            content = content.replace("```json", "")
            content = content.replace("```", "").strip()

        result = json.loads(content)

        sentiment = str(result.get("sentiment", "")).strip().capitalize()
        reason = str(result.get("reason", "")).strip()

        if sentiment not in VALID_LABELS:
            return None, None

        if not reason:
            return None, None

        return sentiment, reason

    except (json.JSONDecodeError, AttributeError, TypeError):
        return None, None

# 4. FUNGSI PELABELAN KOMENTAR
def label_comment(comment):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"Komentar:\n{comment}"
                    }
                ],
                format="json",
                options={
                    "temperature": 0,
                    "num_predict": 120
                }
            )

            raw_response = response["message"]["content"]

            sentiment, reason = parse_response(raw_response)

            if sentiment is not None:
                return sentiment, reason, raw_response

            print(f"Format respons tidak valid. Percobaan {attempt}")

        except Exception as error:
            print(f"Error percobaan {attempt}: {error}")

        time.sleep(2)

    return None, None, None

# 5. FUNGSI MENYIMPAN DATA
def save_progress(df):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

# 6. PROSES UTAMA
def main():

    print("Membaca dataset...")

    df = pd.read_csv(
    INPUT_FILE,
    dtype={"comment_id": str},
    low_memory=False
    )

    if "comment_id" not in df.columns:
        raise ValueError("Kolom comment_id tidak ditemukan.")

    if "comment" not in df.columns:
        raise ValueError("Kolom comment tidak ditemukan.")

    if df["comment_id"].duplicated().any():
        raise ValueError("Terdapat comment_id duplikat.")

    # Memastikan kolom hasil tersedia
    if OUTPUT_FILE.exists():

        print("Memuat hasil pelabelan sebelumnya...")

        previous = pd.read_csv(
            OUTPUT_FILE,
            dtype={"comment_id": str}
        )

        if previous["comment_id"].duplicated().any():
            raise ValueError(
                "Output sebelumnya memiliki comment_id duplikat."
            )

        previous = previous.set_index("comment_id")

        df["sentiment_qwen"] = df["comment_id"].map(
            previous["sentiment_qwen"]
        )

        df["reason_qwen"] = df["comment_id"].map(
            previous["reason_qwen"]
        )

    else:

        df["sentiment_qwen"] = None
        df["reason_qwen"] = None

    # Menentukan komentar yang belum berhasil dilabeli
    pending_indices = df.index[
        ~df["sentiment_qwen"].isin(VALID_LABELS)
    ].tolist()

    total = len(df)

    completed = total - len(pending_indices)

    print("\n========== STATUS DATASET ==========")
    print(f"Total komentar       : {total}")
    print(f"Sudah dilabeli       : {completed}")
    print(f"Belum dilabeli       : {len(pending_indices)}")
    print("====================================\n")

    # PELABELAN OTOMATIS
    for count, idx in enumerate(pending_indices, start=1):

        comment = df.at[idx, "comment"]

        if pd.isna(comment) or not str(comment).strip():

            df.at[idx, "sentiment_qwen"] = "Netral"
            df.at[idx, "reason_qwen"] = "Komentar kosong atau tidak memiliki teks."

        else:

            sentiment, reason, raw_response = label_comment(
                str(comment)
            )

            if sentiment is None:

                print(
                    f"[GAGAL] comment_id: {df.at[idx, 'comment_id']}"
                )

                continue

            df.at[idx, "sentiment_qwen"] = sentiment
            df.at[idx, "reason_qwen"] = reason

        # Menampilkan progres
        if count % 10 == 0:

            print(
                f"Progres: {count}/{len(pending_indices)} "
                f"| Total selesai: "
                f"{df['sentiment_qwen'].isin(VALID_LABELS).sum()}/{total}"
            )

        # Menyimpan setiap 50 komentara
        if count % SAVE_EVERY == 0:

            save_progress(df)

            print("Progres berhasil disimpan.")

    # PENYIMPANAN AKHIR

    save_progress(df)

    # RINGKASAN HASIL
    print("\n========== HASIL PELABELAN ==========")

    print(
        df["sentiment_qwen"].value_counts(dropna=False)
    )

    print("\nTotal komentar:", total)

    print(
        "Berhasil dilabeli:",
        df["sentiment_qwen"].isin(VALID_LABELS).sum()
    )

    print(
        "Belum berhasil:",
        (~df["sentiment_qwen"].isin(VALID_LABELS)).sum()
    )

    print("\nFile tersimpan di:", OUTPUT_FILE)


if __name__ == "__main__":
    main()