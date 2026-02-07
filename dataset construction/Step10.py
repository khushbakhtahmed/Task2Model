import pandas as pd
import re

INPUT_FILE = "Filtered_models_min10rows_collapsed_lower_clean_nondup.csv"
OUTPUT_FILE = "output_clean.csv"
FIRST_COL_NAME = "input_text"  # change if needed

MID_SENTENCE_START_WORDS = {
    "and", "or", "but", "so", "then", "where", "when", "while",
    "however", "thus", "therefore", "hence", "which", "that", "as"
}

def looks_mid_sentence(t: str) -> bool:
    t = t.lstrip()
    if not t:
        return False

    # first word
    first_word = t.split(None, 1)[0]
    # remove leading punctuation from first word
    first_word_clean = re.sub(r"^[,.;:()\"'\-–—]+", "", first_word)

    if not first_word_clean:
        return True

    # if first word is in our "mid-sentence" list
    if first_word_clean.lower() in MID_SENTENCE_START_WORDS:
        return True

    # if first character is lowercase letter
    if first_word_clean[0].islower():
        return True

    return False

def clean_cell(value: str) -> str:
    if pd.isna(value):
        return value

    text = str(value)

    # 3) Remove {{formula:...}} and {{cite:...}} blocks anywhere in the text
    text = re.sub(r"\{\{formula:[^}]*\}\}", "", text)
    text = re.sub(r"\{\{cite:[^}]*\}\}", "", text)

    text = text.strip()

    # 2) Remove starting markers like "— text:", "— bib_entry_raw:", "— caption:"
    text = re.sub(r"^[—\-]?\s*(text|bib_entry_raw|caption)\s*:\s*", "", text).strip()

    # 1) If it *looks* like mid-sentence, drop everything before the first real sentence
    if looks_mid_sentence(text):
        # Try to find a sentence boundary .?! + spaces + Capital
        m = re.search(r"[.?!]\s+([A-Z])", text)
        if m:
            start_idx = m.start(1)
            text = text[start_idx:].lstrip()
        else:
            # fallback: keep after first '.'
            first_dot = text.find(".")
            if first_dot != -1:
                text = text[first_dot + 1:].lstrip()

    # Ensure we end at the last full stop
    last_dot = text.rfind(".")
    if last_dot != -1:
        text = text[:last_dot + 1]

    return text.strip()

def main():
    df = pd.read_csv(INPUT_FILE)

    if FIRST_COL_NAME not in df.columns:
        col0_name = df.columns[0]
    else:
        col0_name = FIRST_COL_NAME

    df[col0_name] = df[col0_name].apply(clean_cell)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Cleaned file written to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
