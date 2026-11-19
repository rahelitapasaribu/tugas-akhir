"""Menguji penerapan kamus normalisasi pada 10.000 komentar YouTube.
Mengganti kata tidak baku berdasarkan kamus normalisasi yang telah disusun.
Membandingkan komentar sebelum dan sesudah normalisasi.
Menghitung frekuensi penggantian kata dan jumlah komentar yang berubah.
Menghasilkan laporan perubahan dan 150 sampel komentar untuk audit.
Hasil pengujian digunakan untuk mengevaluasi kamus sebelum finalisasi. """

import csv
import random
import re
from collections import Counter
from pathlib import Path

# Script disimpan di src/preprocessing/
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / 'data' / 'processed' / 'comments_custom_10000.csv'
DICT_PATH = ROOT / 'data' / 'processed' / 'vocabulary_audit' / 'normalization_dictionary_v3.csv'
OUTPUT_DIR = ROOT / 'data' / 'processed' / 'normalization_test'


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f'File tidak ditemukan: {path}')
    with path.open('r', encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def save_csv(filename, columns, rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / filename).open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    dictionary = read_csv(DICT_PATH)
    comments = read_csv(DATA_PATH)

    if not comments:
        raise ValueError('Dataset komentar kosong.')
    if not dictionary:
        raise ValueError('Kamus normalisasi kosong.')
    if not {'word', 'normalisasi'}.issubset(dictionary[0]):
        raise ValueError('Kamus harus memiliki kolom word dan normalisasi.')
    if not {'comment_id', 'comment'}.issubset(comments[0]):
        raise ValueError('Dataset harus memiliki kolom comment_id dan comment.')

    lookup = {}
    for entry in dictionary:
        key = (entry['word'] or '').strip().casefold()
        value = (entry['normalisasi'] or '').strip()
        if not key or not value:
            raise ValueError(f'Entri kamus kosong: {entry}')
        if key in lookup:
            raise ValueError(f'Duplikasi kata kamus: {key}')
        lookup[key] = entry

    pattern = re.compile(
        r'(?<!\w)(?:' + '|'.join(re.escape(k) for k in sorted(lookup, key=len, reverse=True)) + r')(?!\w)',
        flags=re.IGNORECASE,
    )

    counts = Counter()
    comment_counts = Counter()
    normalized_rows = []
    changes = []
    review_candidates = []
    # Kata berpotensi membutuhkan pemeriksaan khusus, bukan berarti semuanya salah.
    caution_words = {'bundir', 'budir', 'meneh', 'kek', 'team', 'kebaca', 'seneng', 'dah', 'gada', 'sikopet', 'otopsi', 'liat', 'ngajak', 'sampe'}

    for row in comments:
        original = row.get('comment') or ''
        replacements = []

        def replace(match):
            source = match.group(0)
            key = source.casefold()
            target = lookup[key]['normalisasi'].strip()
            if source.isupper() and len(source) > 1:
                target = target.upper()
            elif source[:1].isupper():
                target = target[:1].upper() + target[1:]
            replacements.append((source, target, key))
            counts[key] += 1
            return target

        normalized = pattern.sub(replace, original)
        updated_row = dict(row)
        updated_row['comment_original'] = original
        updated_row['comment_normalized'] = normalized
        updated_row['normalization_applied'] = bool(replacements)
        updated_row['normalization_occurrences'] = len(replacements)
        normalized_rows.append(updated_row)

        if replacements:
            comment_counts.update({key for _, _, key in replacements})
            for source, target, key in replacements:
                changes.append({
                    'comment_id': row['comment_id'],
                    'word_original': source,
                    'word_normalized': target,
                    'dictionary_key': key,
                    'comment_original': original,
                    'comment_normalized': normalized,
                    'source': lookup[key].get('sumber', ''),
                })
            review_candidates.append({
                'comment_id': row['comment_id'],
                'comment_original': original,
                'comment_normalized': normalized,
                'perubahan': '; '.join(f'{source} -> {target}' for source, target, _ in replacements),
                'jumlah_perubahan': len(replacements),
                'perlu_review': 'prioritas' if any(key in caution_words for _, _, key in replacements) else 'sampel',
            })

    report = []
    for entry in dictionary:
        key = entry['word'].strip().casefold()
        report.append({
            'word': entry['word'],
            'normalisasi': entry['normalisasi'],
            'jumlah_kemunculan': counts[key],
            'jumlah_komentar': comment_counts[key],
            'sumber': entry.get('sumber', ''),
            'status_uji': 'terpakai' if counts[key] else 'tidak_muncul',
        })
    report.sort(key=lambda item: (-item['jumlah_kemunculan'], item['word']))

    random.Random(42).shuffle(review_candidates)
    priority = [r for r in review_candidates if r['perlu_review'] == 'prioritas']
    regular = [r for r in review_candidates if r['perlu_review'] != 'prioritas']
    selected_review = priority[:100] + regular[:max(0, 150 - min(100, len(priority)))]

    save_csv('comments_normalized.csv', list(normalized_rows[0]), normalized_rows)
    save_csv('normalization_changes.csv', ['comment_id', 'word_original', 'word_normalized', 'dictionary_key', 'comment_original', 'comment_normalized', 'source'], changes)
    save_csv('normalization_report.csv', ['word', 'normalisasi', 'jumlah_kemunculan', 'jumlah_komentar', 'sumber', 'status_uji'], report)
    save_csv('normalization_review.csv', ['comment_id', 'comment_original', 'comment_normalized', 'perubahan', 'jumlah_perubahan', 'perlu_review'], selected_review)

    print('Total komentar:', len(comments))
    print('Komentar berubah:', len(review_candidates))
    print('Total penggantian:', len(changes))
    print('Pasangan kamus dipakai:', sum(counts[key] > 0 for key in lookup), '/', len(lookup))
    print('Sampel review:', len(selected_review))
    print('Hasil disimpan di:', OUTPUT_DIR)


if __name__ == '__main__':
    main()