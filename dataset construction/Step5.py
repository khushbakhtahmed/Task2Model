
import argparse
import re
import sys
from typing import Optional
import pandas as pd

# --------- Regex patterns ---------
RE_LEADING_HEADERS = re.compile(r'^\s*(?:\[[^\]]+\]\s*){0,5}', flags=re.UNICODE)

RE_STARTING_SECTION_ANY_ORDER = re.compile(
    r'^\s*(?:\[[^\]]+\]\s*)*?\[\s*SECTION\s*:\s*([^\]]+?)\s*\]',
    flags=re.IGNORECASE | re.UNICODE
)

# Remove ANY {{ ... }} token
RE_DOUBLE_BRACE_ANY = re.compile(r'\{\{[^}]+\}\}')

# Standalone REF token
RE_REF_TOKEN = re.compile(r'(?:(?<=\s)|^)[\[\(]*REF[\]\)]*(?=\W|$)', flags=re.IGNORECASE)

# URLs
RE_URL = re.compile(r'https?://\S+|www\.\S+', flags=re.IGNORECASE)

# LaTeX/math
RE_LATEX_DOLLAR = re.compile(r'\$[^$]*\$', flags=re.DOTALL)
RE_LATEX_PARENS = re.compile(r'\\\((?:.|\n)*?\\\)')
RE_LATEX_BRACK  = re.compile(r'\\\[(?:.|\n)*?\\\]')
RE_LATEX_ENV    = re.compile(r'\\begin\{(equation\*?|align\*?|gather\*?)\}.*?\\end\{\1\}', flags=re.DOTALL)
RE_SUBSCRIPT    = re.compile(r'\b([A-Za-z]+)_{\s*[^}]+}', flags=re.UNICODE)
RE_SUPERSCRIPT  = re.compile(r'\b([A-Za-z]+)\^{\s*[^}]+}', flags=re.UNICODE)
RE_NUM_RANGE    = re.compile(r'\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]')

# Model token (any number of guillemets around MODEL)
RE_MODEL_TOKEN = re.compile(r'«+\s*MODEL\s*»+', flags=re.IGNORECASE)

# HuggingFace link containing a MODEL token, with optional leading/trailing guillemets or quotes.
# We capture the whole thing (including surrounding « » or " ") to remove them too.
RE_HF_WITH_MODEL_ANY = re.compile(
    r'(?:[«"]\s*)?https?://huggingface\.co/\s*(?:«\s*MODEL\s*»|<\s*MODEL\s*>)\s*(?:[»"])?',
    flags=re.IGNORECASE
)

HF_PLACEHOLDER = '@@__HF_MODEL_URL__@@'  # protects the link during URL stripping


# --------- Helpers ---------
def detect_starting_section(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = RE_STARTING_SECTION_ANY_ORDER.search(text)
    return m.group(1).strip().lower() if m else None

def strip_leading_headers(text: str) -> str:
    if not isinstance(text, str):
        return text
    return RE_LEADING_HEADERS.sub('', text, count=1)

def normalize_model_and_protect_hf(text: str) -> str:
    """Normalize «MODEL» variants to <MODEL> and protect HF link so it survives URL stripping."""
    if not isinstance(text, str):
        return text
    # 1) Normalize model tokens
    text = text.replace('««MODEL»»', '<MODEL>')
    text = text.replace('«MODEL»', '<MODEL>')
    text = RE_MODEL_TOKEN.sub('<MODEL>', text)
    # 2) Protect any HF link pointing to the model (remove surrounding «», "" if present)
    text = RE_HF_WITH_MODEL_ANY.sub(HF_PLACEHOLDER, text)
    return text

def strip_refs_and_citations(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = RE_DOUBLE_BRACE_ANY.sub('', text)  # remove ANY {{...}}
    text = RE_REF_TOKEN.sub('', text)
    return re.sub(r'\s{2,}', ' ', text).strip()

def remove_urls(text: str) -> str:
    if not isinstance(text, str):
        return text
    return RE_URL.sub('', text)

def remove_latex_math(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = RE_LATEX_ENV.sub('', text)
    text = RE_LATEX_BRACK.sub('', text)
    text = RE_LATEX_PARENS.sub('', text)
    text = RE_LATEX_DOLLAR.sub('', text)
    text = RE_SUBSCRIPT.sub(r'\1', text)
    text = RE_SUPERSCRIPT.sub(r'\1', text)
    text = RE_NUM_RANGE.sub('', text)
    return re.sub(r'\s{2,}', ' ', text).strip()

def restore_protected_urls(text: str) -> str:
    if not isinstance(text, str):
        return text
    if HF_PLACEHOLDER in text:
        # restore as a clean URL without guillemets to avoid truncation issues
        text = text.replace(HF_PLACEHOLDER, 'https://huggingface.co/<MODEL>')
    return text

def preprocess_text(text: str) -> str:
    # 1) Remove structural headers
    text = strip_leading_headers(text)
    # 2) Normalize MODEL + protect HF link
    text = normalize_model_and_protect_hf(text)
    # 3) Remove refs/citations (including ANY {{...}})
    text = strip_refs_and_citations(text)
    # 5) Remove LaTeX/math
    text = remove_latex_math(text)
    # 6) Remove URLs (protected HF link survives)
    text = remove_urls(text)
    # Restore the protected HF link
    text = restore_protected_urls(text)
    # Final whitespace tidy
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


# --------- Main ---------
def main():
    parser = argparse.ArgumentParser(description="Preprocess evaluation data CSV.")
    parser.add_argument("--input", default="models_with10papers_training_data.csv", help="Path to input CSV file")
    parser.add_argument("--output", default="Final_models_with10papers_training_data.csv", help="Path to output CSV file")
    parser.add_argument("--text-col", default="input_text", help="Name of the text column (default: input_text)")
    parser.add_argument("--paper-id-col", default="paper_id", help="Name of the paper ID column (default: paper_id)")
    parser.add_argument("--label-col", default="label", help="Name of the label column (default: label)")
    parser.add_argument("--encoding", default="utf-8", help="CSV encoding (default: utf-8)")
    parser.add_argument("--sep", default=",", help="CSV separator (default: ,)")
    parser.add_argument("--min-words", type=int, default=12, help="Minimum word count after cleaning (default: 5)")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input, encoding=args.encoding, sep=args.sep)
    except Exception as e:
        print(f"Failed to read input CSV: {e}", file=sys.stderr)
        sys.exit(1)

    missing_cols = [c for c in [args.text_col, args.paper_id_col, args.label_col] if c not in df.columns]
    if missing_cols:
        print(f"Missing required columns in CSV: {missing_cols}", file=sys.stderr)
        sys.exit(1)

    # Drop duplicate abstract rows for same (paper_id, label)
    section_col = "__starting_section__"
    df[section_col] = df[args.text_col].apply(detect_starting_section)

    to_drop_idx = set()
    grouped = df.groupby([args.paper_id_col, args.label_col], dropna=False)
    for (pid, lbl), g in grouped:
        sections = set((s or "").strip().lower() for s in g[section_col].fillna(""))
        if "abstract.text" in sections and "metadata.abstract" in sections:
            idxs = g.index[g[section_col].str.lower().str.strip() == "metadata.abstract"].tolist()
            to_drop_idx.update(idxs)

    before = len(df)
    if to_drop_idx:
        df = df.drop(index=list(to_drop_idx))

    # Apply preprocessing
    df[args.text_col] = df[args.text_col].astype(str).map(preprocess_text)

    # Drop too-short rows
    word_counts = df[args.text_col].str.split().apply(len)
    short_idx = word_counts[word_counts < args.min_words].index
    df = df.drop(index=short_idx)

    # Cleanup
    if section_col in df.columns:
        df = df.drop(columns=[section_col])

    try:
        df.to_csv(args.output, index=False, encoding=args.encoding)
    except Exception as e:
        print(f"Failed to write output CSV: {e}", file=sys.stderr)
        sys.exit(1)

    after = len(df)
    print("Preprocessing complete.")
    print(f"Rows input:  {before}")
    print(f"Rows output: {after}")
    print(f"Dropped metadata.abstract rows: {len(to_drop_idx)}")
    print(f"Dropped short rows (<{args.min_words} words): {len(short_idx)}")
    print(f"Saved to: {args.output}")

if __name__ == "__main__":
    main()
