#!/usr/bin/env python3
"""Filter rows in a CSV keeping only those whose `input_text` contains
at least N letters/digits (alphanumeric characters).

Usage:
  python3 count_short_rows.py
  python3 count_short_rows.py --infile input.csv --outfile out.csv --min 200
"""

import argparse
import csv
import sys


def count_alnum(s: str) -> int:
    if not s:
        return 0
    return sum(1 for c in s if c.isalnum())


def filter_file(infile: str, outfile: str, min_count: int, input_col: str) -> None:
    kept = 0
    total = 0
    try:
        with open(infile, newline='', encoding='utf-8') as inf, open(outfile, 'w', newline='', encoding='utf-8') as outf:
            reader = csv.DictReader(inf)
            if reader.fieldnames is None:
                print('ERROR: input file has no header', file=sys.stderr)
                sys.exit(2)
            if input_col not in reader.fieldnames:
                print(f"ERROR: column '{input_col}' not found in input. Available: {reader.fieldnames}", file=sys.stderr)
                sys.exit(2)
            writer = csv.DictWriter(outf, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                total += 1
                text = row.get(input_col, '') or ''
                if count_alnum(text) >= min_count:
                    writer.writerow(row)
                    kept += 1
    except FileNotFoundError:
        print(f"ERROR: file not found: {infile}", file=sys.stderr)
        sys.exit(2)

    print(f"Processed {total} rows, kept {kept} rows")


def main():
    p = argparse.ArgumentParser(description='Filter CSV rows by alphanumeric character count in input_text')
    p.add_argument('--infile', '-i',
                   default='output_clean.csv',
                   help='Input CSV path (from sentence-boundary cleanup)')
    p.add_argument('--outfile', '-o',
                   default='output_clean_min200chars.csv',
                   help='Output CSV path')
    p.add_argument('--min', '-m', type=int, default=200, dest='min_count',
                   help='Minimum number of letters/digits to keep (e.g., 200, 300, 500)')
    p.add_argument('--col', '-c', default='input_text', help='Column name containing text')
    args = p.parse_args()

    filter_file(args.infile, args.outfile, args.min_count, args.col)


if __name__ == '__main__':
    main()
