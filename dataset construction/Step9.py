#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
import sys
from typing import Optional

csv.field_size_limit(sys.maxsize)

_SENTENCE_SPLIT_RE = re.compile(r'(?:(?<=[\.\!\?]["’”"])|(?<=[\.\!\?]))\s+')

_LEADING_PREFIXES = [
    r'^\s*[\-\—\–]\s*bib_entry_raw:?',
    r'^\s*bib_entry_raw:?',
    r'^\s*[\-\—\–]\s*text:?',
    r'^\s*text:?',
    r'^\s*[-]{1,3}\s*',
    r'^\s*[\u2013\u2014]\s*',
]

def remove_leading_prefixes(s: str) -> str:
    if not s:
        return s
    for pat in _LEADING_PREFIXES:
        s = re.sub(pat, '', s, flags=re.IGNORECASE)
    return s.lstrip()

def first_alpha_is_upper(sent: str) -> bool:
    m = re.search(r'[A-Za-z]', sent)
    return bool(m and m.group(0).isupper())

def ends_with_sentence_punct(sent: str) -> bool:
    sent = sent.rstrip()
    return bool(sent) and sent[-1] in '.!?'

def split_into_sentences(s: str):
    s = re.sub(r'\s+', ' ', s).strip()
    if not s:
        return []
    parts = _SENTENCE_SPLIT_RE.split(s)
    return [p.strip() for p in parts if p.strip()]

def clean_text_field(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None

    s = remove_leading_prefixes(s)
    s = s.replace('\n', ' ').replace('\r', ' ')
    sentences = split_into_sentences(s)
    if not sentences:
        return None

    while sentences and not first_alpha_is_upper(sentences[0]):
        sentences.pop(0)

    while sentences and not ends_with_sentence_punct(sentences[-1]):
        sentences.pop()

    if not sentences:
        return None

    cleaned = ' '.join(sentences).strip()
    cleaned = cleaned.lower()
    return cleaned if cleaned else None

def process_csv(input_path: str, output_path: str, column: str, drop_empty: bool = True, encoding: str = 'utf-8'):
    processed = 0
    dropped = 0
    changed = 0

    with open(input_path, 'r', newline='', encoding=encoding) as inf:
        reader = csv.DictReader(inf)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError('Input CSV has no header')
        if column not in fieldnames:
            raise ValueError(f"Column '{column}' not found in input CSV headers: {fieldnames}")

        rows_out = []
        for row in reader:
            processed += 1
            orig = row.get(column, '')
            cleaned = clean_text_field(orig)

            if cleaned is None or cleaned == '':
                dropped += 1
                if not drop_empty:
                    row[column] = ''
                    rows_out.append(row)
            else:
                if cleaned != (orig or '').strip().lower():
                    changed += 1
                row[column] = cleaned
                rows_out.append(row)

    with open(output_path, 'w', newline='', encoding=encoding) as outf:
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    print(f"Processed {processed} rows. Kept {len(rows_out)} rows; dropped {dropped}. Modified {changed} rows.")

def main():
    parser = argparse.ArgumentParser(
        description='Clean and lowercase CSV `input_text` column.'
    )
    parser.add_argument(
        '--input', '-i',
        default='Filtered_models_min10rows_collapsed_nondup.csv',
        help='Input CSV file path'
    )
    parser.add_argument(
        '--output', '-o',
        default='Filtered_models_min10rows_collapsed_lower_clean_nondup.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--column', '-c',
        default='input_text',
        help='Column name to clean (default: input_text)'
    )
    parser.add_argument(
        '--keep-empty',
        action='store_true',
        help='Keep rows that become empty after cleaning (default: drop them)'
    )
    parser.add_argument('--encoding', default='utf-8', help='File encoding')
    args = parser.parse_args()

    process_csv(
        args.input,
        args.output,
        args.column,
        drop_empty=not args.keep_empty,
        encoding=args.encoding
    )
    print(f"Output written to: {args.output}")

if __name__ == '__main__':
    main()
