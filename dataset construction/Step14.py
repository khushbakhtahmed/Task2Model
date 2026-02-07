#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from collections import Counter

HERE = Path(__file__).resolve().parent
INPUT = HERE / "output_clean_with_paper_id_min200chars_corrected_with_family_mapped_all.csv"
OUTPUT = HERE / "output_clean_with_paper_id_min200chars_corrected_with_family_mapped_all_filtered.csv"

EXCLUDE = {"turbo", "test1", "test2", "testmodel", "glider", "ppo2", "factcc", "factkb"}
CHUNK = 10000

# helper to choose collapsed label column
def choose_collapsed_col(cols):
    lower = [c.lower() for c in cols]
    for candidate in ["collapsed label", "collapsed_label", "collapsed", "label", "class"]:
        if candidate in lower:
            return cols[lower.index(candidate)]
    return None

if not INPUT.exists():
    print(f"Input file not found: {INPUT}")
    raise SystemExit(1)

# read header
df0 = pd.read_csv(INPUT, nrows=0)
cols = list(df0.columns)
collapsed_col = choose_collapsed_col(cols)
if collapsed_col is None:
    print("Could not determine collapsed label column. Columns:", cols)
    raise SystemExit(1)

# detect family column
family_col = None
for c in cols:
    if c.lower() in ("family label", "family_label", "family"):
        family_col = c
        break
if family_col is None:
    # we still proceed but family counts may be unavailable
    print("Warning: family column not found. Expected 'family label'.")

# detect 'label' column for unique label count
label_col = None
for c in cols:
    if c.lower() in ("label", "model", "model id", "model_id", "class"):
        label_col = c
        break

# process in chunks
first = True
total_kept = 0
unique_collapsed = set()
unique_family = set()
unique_label = set()

reader = pd.read_csv(INPUT, chunksize=CHUNK, dtype=str)
for chunk in reader:
    # normalize collapsed values for matching
    chunk[collapsed_col] = chunk[collapsed_col].fillna("")
    mask = ~chunk[collapsed_col].str.strip().str.lower().isin(EXCLUDE)
    kept = chunk[mask]
    if kept.empty:
        continue
    # update stats
    total_kept += len(kept)
    unique_collapsed.update(kept[collapsed_col].astype(str).str.strip().unique().tolist())
    if family_col and family_col in kept.columns:
        unique_family.update(kept[family_col].astype(str).str.strip().unique().tolist())
    if label_col and label_col in kept.columns:
        unique_label.update(kept[label_col].astype(str).str.strip().unique().tolist())

    if first:
        kept.to_csv(OUTPUT, index=False, mode="w")
        first = False
    else:
        kept.to_csv(OUTPUT, index=False, header=False, mode="a")

# print report
print(f"Filtered output written to: {OUTPUT}")
print(f"Rows after filtering: {total_kept}")
print(f"Unique collapsed_label: {len(unique_collapsed)}")
if label_col:
    print(f"Unique label ({label_col}): {len(unique_label)}")
else:
    print("Unique label: column not detected")
if family_col:
    print(f"Unique family_label ({family_col}): {len(unique_family)}")
else:
    print("Unique family_label: column not detected")

# save a small report json
import json
report = {
    "rows": int(total_kept),
    "unique_collapsed_label": int(len(unique_collapsed)),
    "unique_label_col": label_col or None,
    "unique_label_count": int(len(unique_label)) if label_col else None,
    "family_col": family_col or None,
    "unique_family_count": int(len(unique_family)) if family_col else None,
}
(HERE / "filter_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("Saved report to filter_report.json")
