# Proses ulang hanya kata berstatus gagal, tanpa menghapus hasil audit sebelumnya.

import json
import os
import time
from pathlib import Path

import pandas as pd
import ollama

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data' / 'processed' / 'vocabulary_audit'
INPUT_FILE = DATA_DIR / 'usulan_qwen_prioritas_1376.csv'
OUTPUT_FILE = DATA_DIR / 'usulan_qwen_prioritas_1376_retry.csv'
MODEL_NAME = 'qwen2.5:3b'
BATCH_SIZE = 3
MAX_RETRIES = 3

SYSTEM_PROMPT = '''Kamu mengevaluasi satu per satu kata target dalam komentar YouTube berbahasa Indonesia.
Analisis hanya kata target. Kata lain di contoh komentar bukan subjek klasifikasi.
Keputusan: normalisasi (padanan pasti tanpa mengubah makna), pertahankan (kata baku, kata asing valid, nama orang, ekspresi sentiment), review_manual (ambigu).
Jangan menciptakan bentuk kata, menerjemahkan kata asing yang valid, atau mengubah negasi dan intensitas.
Contoh: ni -> ini (jika bermakna ini), skrng -> sekarang, gk -> tidak, soohyun -> pertahankan, ago -> pertahankan, air -> pertahankan, skip -> pertahankan.
Jawab JSON saja: {"results":[{"word":"...", "decision":"normalisasi", "suggested_normalization":"...", "reason":"..."}]}
Harus tepat satu hasil per kata target, mempertahankan ejaan kata target secara identik.
Untuk pertahankan/review_manual, suggested_normalization harus kosong.
'''

VALID_DECISIONS = {'normalisasi', 'pertahankan', 'review_manual'}


def save_atomic(df):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_FILE.with_suffix('.tmp')
    df.to_csv(tmp, index=False, encoding='utf-8-sig')
    os.replace(tmp, OUTPUT_FILE)


def qwen_request(rows):
    payload = []
    for row in rows.itertuples(index=False):
        payload.append({
            'word': str(row.word),
            'frequency': int(row.frequency),
            'example': str(row.examples_readable)[:450],
        })

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)},
        ],
        format='json',
        options={'temperature': 0},
    )
    body = json.loads(response['message']['content'])
    if not isinstance(body, dict) or not isinstance(body.get('results'), list):
        raise ValueError('Qwen tidak mengembalikan results berbentuk list.')

    wanted = set(rows['word'].astype(str))
    parsed = {}
    for item in body['results']:
        if not isinstance(item, dict):
            continue
        word = str(item.get('word', '')).strip()
        if word not in wanted or word in parsed:
            continue
        decision = str(item.get('decision', '')).strip().lower()
        suggestion = str(item.get('suggested_normalization') or '').strip().lower()
        reason = str(item.get('reason') or '').strip()
        if decision not in VALID_DECISIONS:
            continue
        if decision == 'normalisasi' and (not suggestion or suggestion == word.lower()):
            decision = 'review_manual'
            suggestion = ''
            reason = 'Saran normalisasi kosong/sama seperti kata asal.'
        if decision != 'normalisasi':
            suggestion = ''
        parsed[word] = (decision, suggestion, reason)

    if set(parsed) != wanted:
        raise ValueError(f'Qwen melewatkan {len(wanted - set(parsed))} kata.')
    return parsed


def retry(rows):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return qwen_request(rows)
        except Exception as exc:
            print(f'  Percobaan {attempt}/{MAX_RETRIES} gagal: {exc}')
            if attempt < MAX_RETRIES:
                time.sleep(1)
    return None


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f'File tidak ditemukan: {INPUT_FILE}')

    base = pd.read_csv(INPUT_FILE, keep_default_na=False)
    required = {'word', 'frequency', 'examples_readable', 'status_qwen',
                'usulan_keputusan', 'usulan_normalisasi', 'alasan_usulan'}
    missing = required - set(base.columns)
    if missing:
        raise ValueError(f'Kolom tidak ditemukan: {sorted(missing)}')
    if base.columns.duplicated().any() or base['word'].duplicated().any():
        raise ValueError('Ada nama kolom atau kata duplikat. Periksa input.')

    # Output terpisah menjaga file asli dan semua isian validasi manusia.
    if OUTPUT_FILE.exists():
        output = pd.read_csv(OUTPUT_FILE, keep_default_na=False)
        if list(output['word']) != list(base['word']):
            raise ValueError('Checkpoint berbeda dari input. Jangan digabung otomatis.')
    else:
        output = base.copy()
        save_atomic(output)

    failed = output[output['status_qwen'].eq('gagal')]
    print(f'Total: {len(output)} | Sudah berhasil: {(output.status_qwen == "berhasil").sum()} | Sisa gagal: {len(failed)}')

    words = failed['word'].tolist()
    for start in range(0, len(words), BATCH_SIZE):
        group = words[start:start + BATCH_SIZE]
        rows = output[output['word'].isin(group)]
        result = retry(rows)

        # Jika batch gagal, setiap kata dicoba sendiri agar tidak kehilangan satu batch.
        if result is None:
            result = {}
            for word in group:
                single = output[output['word'].eq(word)]
                one = retry(single)
                if one:
                    result.update(one)

        for word, (decision, suggestion, reason) in result.items():
            idx = output.index[output['word'].eq(word)][0]
            output.at[idx, 'usulan_keputusan'] = decision
            output.at[idx, 'usulan_normalisasi'] = suggestion
            output.at[idx, 'alasan_usulan'] = reason
            output.at[idx, 'status_qwen'] = 'berhasil'

        save_atomic(output)
        print(f'Batch {start // BATCH_SIZE + 1}: berhasil {len(result)}/{len(group)} | masih gagal {(output.status_qwen == "gagal").sum()}')

    print(f'\nFile hasil: {OUTPUT_FILE}')
    print(output['status_qwen'].value_counts().to_string())
    print('\nUsulan keputusan:')
    print(output['usulan_keputusan'].value_counts().to_string())
    print('Semua usulan Qwen masih perlu divalidasi sebelum masuk kamus final.')

if __name__ == '__main__':
    main()