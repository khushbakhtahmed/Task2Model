#!/usr/bin/env python
import json
from pathlib import Path
import re
import numpy as np
from collections import defaultdict
import pandas as pd

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
HERE = Path(__file__).resolve().parent

# Input CSV (adjust name if needed)
INPUT_CSV = HERE / "output_clean_with_paper_id_min200chars_corrected.csv"

# Output CSV will be this file (same folder)
OUTPUT_CSV = HERE / "output_clean_with_paper_id_min200chars_corrected_with_family.csv"

# how many rows to process per chunk
CHUNK_SIZE = 10000

# ---------------------------------------------------------
# Collapsed label  ->  family
# (keys are exactly the values from classes.csv, lower-case)
# ---------------------------------------------------------
label_to_family = {
    "all-minilm-l6-v2": "embeddings",
    "all-mpnet-base-v2": "embeddings",
    "alpaca-7b": "alpaca",
    "bart-base": "bart",
    "bart-large": "bart",
    "bert-base": "bert",
    "bert-base-cased": "bert",
    "bert-base-uncased": "bert",
    "biogpt": "biogpt",
    "bloom": "bloom",
    "bloomz": "bloom",
    "bm25": "bm25",
    "clinicalbert": "bert",
    "codellama-7b": "code",
    "codeparrot": "code",
    "dbpedia": "dbpedia",
    "deepseek-v2": "deepseek",
    "falcon-40b": "falcon",
    "falcon-7b": "falcon",
    "finbert": "bert",
    "flan-t5-base": "t5",
    "flan-t5-large": "t5",
    "flan-t5-xl": "t5",
    "flan-t5-xxl": "t5",
    "flan-ul2": "t5",
    "gemini": "gemini",
    "gemma-2b": "gemma",
    "gemma-7b": "gemma",
    "gpt-j-6b": "gpt",
    "gpt-neox-20b": "gpt",
    "gpt2": "gpt",
    "gpt2-large": "gpt",
    "gpt2-medium": "gpt",
    "gpt2-xl": "gpt",
    "gsm8k": "gsm8k",
    "hatebert": "bert",
    "hebert": "bert",
    "labse": "embeddings",
    "llama-13b": "llama",
    "llama-2-13b": "llama",
    "llama-2-70b-chat": "llama",
    "llama-2-7b": "llama",
    "llama-2-7b-chat": "llama",
    "llama-2-7b-chat-hf": "llama",
    "llama-3-8b": "llama",
    "llama-3-8b-instruct": "llama",
    "llama-3.1-70b": "llama",
    "llama-3.1-8b": "llama",
    "llama-3.1-8b-instruct": "llama",
    "llama-65b": "llama",
    "llama-7b": "llama",
    "meta-llama-3-8b-instruct": "llama",
    "metamodel": "metamodel",
    "mistral-7b": "mistral",
    "mistral-7b-instruct-v0.2": "mistral",
    "mistral-7b-v0.1": "mistral",
    "mpt-7b": "mpt",
    "opt-1.3b": "opt",
    "opt-125m": "opt",
    "opt-13b": "opt",
    "opt-2.7b": "opt",
    "opt-30b": "opt",
    "opt-6.7b": "opt",
    "opt-66b": "opt",
    "phi-1": "phi",
    "phi-2": "phi",
    "qwen-7b": "qwen",
    "qwen-audio": "qwen",
    "qwen-vl": "qwen",
    "qwen-vl-chat": "qwen",
    "qwen2-7b": "qwen",
    "qwen2-7b-instruct": "qwen",
    "qwen2.5": "qwen",
    "roberta-base": "roberta",
    "roberta-large": "roberta",
    "santacoder": "code",
    "starcoder": "code",
    "t5-11b": "t5",
    "t5-3b": "t5",
    "t5-base": "t5",
    "t5-large": "t5",
    "t5-small": "t5",
    "ties-merging": "ties-merging",
    "tinyllama": "llama",
    "turbo": "turbo",
    "vicuna-7b": "vicuna",
    "vicuna-7b-v1.5": "vicuna",
    "xlm-roberta-base": "roberta",
    "xlm-roberta-large": "roberta",
}

# ---------------------------------------------------------
# Core computation
# ---------------------------------------------------------
def map_collapsed_label_to_family_series(series, label_to_family):
    """Map a pandas Series of collapsed labels to family labels.

    Returns the mapped Series and a set of unmapped unique labels.
    """
    unmapped = set()

    def _map_val(v):
        if pd.isna(v):
            return "unknown"
        key = str(v).strip().lower()
        if key in label_to_family:
            return label_to_family[key]
        unmapped.add(key)
        return "unknown"

    mapped = series.map(_map_val)
    return mapped, unmapped


def auto_map_labels(unmapped_list):
    """Heuristically map unmapped collapsed labels to families based on tokens.

    Returns a dict mapping collapsed_label -> family for labels we can auto-resolve.
    """
    token_to_family = {
        "bert": "bert",
        "roberta": "roberta",
        "gpt": "gpt",
        "gpt-": "gpt",
        "llama": "llama",
        "meta-llama": "llama",
        "vicuna": "vicuna",
        "mistral": "mistral",
        "falcon": "falcon",
        "opt": "opt",
        "t5": "t5",
        "flan": "t5",
        "mpt": "mpt",
        "gemma": "gemma",
        "gemini": "gemini",
        "phi": "phi",
        "qwen": "qwen",
        "bloom": "bloom",
        "biogpt": "biogpt",
        "codellama": "code",
        "code": "code",
        "starcoder": "code",
        "santacoder": "code",
        "labse": "embeddings",
        "minilm": "embeddings",
        "mpnet": "embeddings",
        "bm25": "bm25",
        "deepseek": "deepseek",
        "turbo": "turbo",
        "alpaca": "alpaca",
        "gsm8k": "gsm8k",
    }

    out = {}
    for label in unmapped_list:
        l = (label or "").lower()
        # strip common suffixes/prefixes and version tokens
        l_clean = l
        # normalize by replacing non-alphanumeric runs with a dash
        l_clean = re.sub(r"[^a-z0-9]+", "-", l_clean)
        parts = [p for p in l_clean.split("-") if p]
        mapped = None
        # check full token presence
        for tkn, fam in token_to_family.items():
            if tkn in l:
                mapped = fam
                break

        # otherwise inspect parts
        if mapped is None:
            for p in parts:
                if p in token_to_family:
                    mapped = token_to_family[p]
                    break

        # fallback: use first part as family if nothing matched
        if mapped is None and parts:
            mapped = parts[0]

        if mapped:
            out[label] = mapped

    return out

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def _choose_label_column(sample_df):
    # prefer explicit names
    cols = list(sample_df.columns)
    lower_cols = [c.lower() for c in cols]
    for candidate in ["collapsed label", "collapsed_label", "collapsed", "label", "class"]:
        if candidate in lower_cols:
            return cols[lower_cols.index(candidate)]

    # otherwise pick the column with highest fraction of values matching known labels
    best_col = None
    best_score = 0.0
    keys = set(label_to_family.keys())
    for c in cols:
        ser = sample_df[c].astype(str).str.strip().str.lower()
        non_null = ser[ser != "nan"]
        if len(non_null) == 0:
            continue
        unique_vals = pd.Series(non_null.unique())
        matches = unique_vals.isin(list(keys)).sum()
        score = matches / float(len(unique_vals))
        if score > best_score:
            best_score = score
            best_col = c

    return best_col


def main():
    if not INPUT_CSV.exists():
        print(f"Input CSV not found: {INPUT_CSV}")
        return

    print(f"Reading header/sample from {INPUT_CSV}...")
    sample = pd.read_csv(INPUT_CSV, nrows=2000, dtype=str)
    label_col = _choose_label_column(sample)
    if label_col is None:
        print("Could not auto-detect a collapsed label column. Please check the CSV headers.")
        print("Columns:", list(sample.columns))
        return

    print(f"Using column '{label_col}' as collapsed label column.")

    unmapped_all = set()
    collapsed_to_family = {}
    first = True

    reader = pd.read_csv(INPUT_CSV, dtype=str, chunksize=CHUNK_SIZE)
    written = 0
    for chunk in reader:
        mapped_series, unmapped = map_collapsed_label_to_family_series(chunk[label_col], label_to_family)
        chunk["family label"] = mapped_series

        # collect mapping for unique collapsed values in this chunk
        for orig_val, fam in zip(chunk[label_col].fillna("").astype(str), chunk["family label"]):
            key = orig_val.strip().lower()
            if key not in collapsed_to_family:
                collapsed_to_family[key] = fam

        unmapped_all.update(unmapped)

        # write
        if first:
            chunk.to_csv(OUTPUT_CSV, index=False, mode="w")
            first = False
        else:
            chunk.to_csv(OUTPUT_CSV, index=False, header=False, mode="a")

        written += len(chunk)

    # save mappings and unmapped
    mapping_path = HERE / "collapsed_to_family_mapping.json"
    unmapped_path = HERE / "unmapped_collapsed_labels.json"
    mapping_clean = {k: v for k, v in collapsed_to_family.items()}
    mapping_path.write_text(json.dumps(mapping_clean, indent=2), encoding="utf-8")
    unmapped_list = sorted(list(u for u in unmapped_all if u != ""))
    unmapped_path.write_text(json.dumps(unmapped_list, indent=2), encoding="utf-8")

    print(f"Wrote {written} rows to {OUTPUT_CSV}")
    print(f"Saved mapping to {mapping_path}")
    print(f"Saved unmapped labels ({len(unmapped_list)}) to {unmapped_path}")

    # Attempt to auto-map unmapped labels using heuristics
    if len(unmapped_list) > 0:
        auto_map = auto_map_labels(unmapped_list)
        if auto_map:
            print(f"Auto-mapped {len(auto_map)} labels. Re-writing final CSV with these mappings.")
            # extend label_to_family with auto mappings and re-run mapping to create final output
            label_to_family.update(auto_map)
            FINAL_OUTPUT_CSV = HERE / "output_clean_with_paper_id_min200chars_correctedAHMED_with_family_mapped_all.csv"
            first2 = True
            written2 = 0
            reader2 = pd.read_csv(INPUT_CSV, dtype=str, chunksize=CHUNK_SIZE)
            for chunk in reader2:
                mapped_series, _ = map_collapsed_label_to_family_series(chunk[label_col], label_to_family)
                chunk["family label"] = mapped_series
                if first2:
                    chunk.to_csv(FINAL_OUTPUT_CSV, index=False, mode="w")
                    first2 = False
                else:
                    chunk.to_csv(FINAL_OUTPUT_CSV, index=False, header=False, mode="a")
                written2 += len(chunk)

            # save updated mappings
            updated_mapping_path = HERE / "collapsed_to_family_mapping_auto.json"
            updated_mapping_path.write_text(json.dumps({k: v for k, v in collapsed_to_family.items()}, indent=2), encoding="utf-8")
            print(f"Wrote {written2} rows to {FINAL_OUTPUT_CSV}")
            print(f"Saved auto mapping to {updated_mapping_path}")
        else:
            print("No auto-mappings found for unmapped labels. Please review unmapped_collapsed_labels.json")

        # --- produce unique collapsed -> family CSV
        print("Generating unique collapsed label -> family CSV with counts...")
        counts = defaultdict(int)
        reader3 = pd.read_csv(INPUT_CSV, dtype=str, chunksize=CHUNK_SIZE)
        for chunk in reader3:
            if label_col in chunk.columns:
                vals = chunk[label_col].fillna("").astype(str).str.strip()
                for v in vals:
                    counts[v] += 1

        unique_list = sorted(list(counts.keys()))
        rows = []
        # prepare mapping dict: normalize keys
        norm_map = {k: v for k, v in label_to_family.items()}
        # attempt auto map for labels not in norm_map
        to_try = [u for u in unique_list if u.strip().lower() not in norm_map]
        auto = auto_map_labels([u.strip().lower() for u in to_try]) if to_try else {}

        for u in unique_list:
            key = u.strip().lower()
            fam = norm_map.get(key)
            if not fam:
                fam = auto.get(key)
            if not fam:
                # fallback to first token
                parts = [p for p in re.split(r"[^a-z0-9]+", key) if p]
                fam = parts[0] if parts else "unknown"
            rows.append({"collapsed_label": u, "family_label": fam, "count": counts.get(u, 0)})

        out_df = pd.DataFrame(rows)
        unique_csv = HERE / "unique_collapsed_to_family.csv"
        out_df.to_csv(unique_csv, index=False)
        print(f"Wrote {len(rows)} unique mappings (with counts) to {unique_csv}")


if __name__ == "__main__":
    main()
