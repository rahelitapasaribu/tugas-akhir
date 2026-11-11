# Audit ulang kosakata frekuensi 5–99 menggunakan Qwen lokal.

import json
import os
import time
from pathlib import Path

import ollama
import pandas as pd

# Lokasi file proyek
BASE_DIR = Path(__file__).resolve().parents[2] if 'src' in Path(__file__).parts else Path.cwd()
INPUT_FILE = BASE_DIR / 'data' / 'processed' / 'vocabulary_audit' / 'prioritas_audit_frekuensi_5_99.csv'
OUTPUT_FILE = BASE_DIR / 'data' / 'processed' / 'vocabulary_audit' / 'usulan_qwen_prioritas_1376.csv'
MODEL_NAME = 'qwen2.5:3b'
BATCH_SIZE = 5
MAX_RETRIES = 3

# Instruksi agar model tidak mengarang bentuk kata dan tidak mengubah sentimen
SYSTEM_PROMPT = '''Kamu adalah pemeriksa kosakata komentar YouTube berbahasa Indonesia.
Analisis setiap KATA TARGET secara individual. Konteks hanya membantu memahami kata target;
jangan mengklasifikasikan berdasarkan nama orang yang kebetulan muncul di komentarnya.
Keputusan yang tersedia:
- normalisasi: singkatan/typo/slang dengan pengganti yang sangat jelas dan makna sama.
- pertahankan: kata yang sudah tepat, nama entitas, kata asing valid, istilah khusus,
  atau ekspresi penting untuk sentimen.
- review_manual: kata ambigu atau tidak yakin.
Jangan menebak. Jangan menghilangkan negasi, intensitas emosi, atau mengubah arti.
Contoh: 'ganti' -> pertahankan; 'membela' -> pertahankan;
'skrng' -> normalisasi 'sekarang'; 'gk' -> normalisasi 'tidak';
'ksh' -> pertahankan; 'ago' -> pertahankan; 'dr' -> review_manual jika konteks tidak jelas.
Jawab HANYA dalam JSON valid:
{"results": [{"word":"...", "decision":"normalisasi|pertahankan|review_manual",
"suggested_normalization":"...", "reason":"alasan singkat"}]}
Wajib persis satu hasil untuk setiap kata yang diberikan. Jangan tambahkan kata lain.
Jika bukan normalisasi, suggested_normalization harus berupa string kosong.
'''


def save_csv(df):
    # Simpan secara atomik agar hasil tidak rusak jika proses berhenti
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_file = OUTPUT_FILE.with_suffix('.tmp')
    df.to_csv(temp_file, index=False, encoding='utf-8-sig')
    os.replace(temp_file, OUTPUT_FILE)


def ask_qwen(batch):
    payload = [
        {
            'word': str(row.word),
            'frequency': int(row.frequency),
            'contoh_komentar': str(row.examples_readable)[:500],
        }
        for row in batch.itertuples(index=False)
    ]
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)},
        ],
        format='json',
        options={'temperature': 0},
    )
    raw = json.loads(response['message']['content'])
    if not isinstance(raw, dict) or not isinstance(raw.get('results'), list):
        raise ValueError('Format JSON hasil tidak valid')
    results = raw['results']
    expected = set(batch['word'].astype(str))
    observed = [str(item.get('word', '')) for item in results if isinstance(item, dict)]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError('Qwen melewatkan/menggandakan kata dalam batch')

    accepted = []
    for item in results:
        word = str(item['word'])
        decision = str(item.get('decision', '')).strip().lower()
        suggestion = str(item.get('suggested_normalization') or '').strip().lower()
        reason = str(item.get('reason') or '').strip()
        if decision not in {'normalisasi', 'pertahankan', 'review_manual'}:
            raise ValueError(f'Keputusan tidak valid: {word}')
        if decision == 'normalisasi' and (not suggestion or suggestion == word.lower()):
            decision, suggestion = 'review_manual', ''
            reason = 'Saran model kosong atau sama dengan kata asal; perlu validasi.'
        if decision != 'normalisasi':
            suggestion = ''
        accepted.append({
            'word': word,
            'usulan_keputusan': decision,
            'usulan_normalisasi': suggestion,
            'alasan_usulan': reason,
            'status_qwen': 'berhasil',
        })
    return accepted


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f'Input tidak ditemukan: {INPUT_FILE}')
    source = pd.read_csv(INPUT_FILE, keep_default_na=False)
    needed = {'word', 'frequency', 'examples_readable'}
    if not needed.issubset(source.columns):
        raise ValueError(f'Kolom input kurang: {needed - set(source.columns)}')
    source['word'] = source['word'].astype(str).str.strip()
    source = source.drop_duplicates('word').copy()

    # Muat checkpoint jika sudah ada; usulan yang sukses tidak diproses ulang
    if OUTPUT_FILE.exists():
        result = pd.read_csv(OUTPUT_FILE, keep_default_na=False)
        valid_completed = set(result.loc[result['status_qwen'] == 'berhasil', 'word'])
        result = result[result['word'].isin(valid_completed)].copy()
    else:
        result = pd.DataFrame(columns=list(dict.fromkeys(list(source.columns) + [
            'usulan_keputusan', 'usulan_normalisasi', 'alasan_usulan',
            'status_qwen', 'keputusan_disetujui', 'normalisasi_disetujui', 'catatan_validasi'
        ])))
        valid_completed = set()

    remaining = source[~source['word'].isin(valid_completed)].copy()
    total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f'Kata input: {len(source)} | Sudah berhasil: {len(valid_completed)} | Sisa: {len(remaining)}')

    for start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining.iloc[start:start + BATCH_SIZE].copy()
        batch_no = (start // BATCH_SIZE) + 1
        suggestions = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                suggestions = ask_qwen(batch)
                break
            except Exception as error:
                print(f'Batch {batch_no}, percobaan {attempt}: {error}')
                time.sleep(1)

        if suggestions is None:
            suggestions = [
                {
                    'word': str(word), 'usulan_keputusan': 'review_manual',
                    'usulan_normalisasi': '', 'alasan_usulan': 'Qwen gagal setelah retry.',
                    'status_qwen': 'gagal',
                }
                for word in batch['word']
            ]

        additions = pd.DataFrame(suggestions)
        additions = batch.merge(additions, on='word', how='left', validate='one_to_one')
        additions['keputusan_disetujui'] = ''
        additions['normalisasi_disetujui'] = ''
        additions['catatan_validasi'] = ''
        result = pd.concat([result, additions], ignore_index=True)
        result = result.drop_duplicates('word', keep='last')
        save_csv(result)
        print(f'Batch {batch_no}/{total_batches} selesai. Total tersimpan: {len(result)}')

    print(f'\nHasil: {OUTPUT_FILE}')
    print(result['usulan_keputusan'].value_counts(dropna=False).to_string())
    print('Periksa usulan sebelum mengisi kolom keputusan_disetujui.')


if __name__ == '__main__':
    main()