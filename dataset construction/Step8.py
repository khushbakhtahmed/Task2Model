
import pandas as pd
from pathlib import Path

# --------------------
# Paths
# --------------------
THIS_DIR = Path(__file__).resolve().parent
DATA_IN = THIS_DIR / "Filtered_models_min10rows_collapsed.csv"
DATA_OUT = THIS_DIR / "Filtered_models_min10rows_collapsed_nondup.csv"

# --------------------
# Load data
# --------------------
df = pd.read_csv(DATA_IN)

# --------------------
# Remove duplicates
# --------------------
df_nondup = df.drop_duplicates(subset=["collapsed_label", "input_text"])

# --------------------
# Save cleaned CSV
# --------------------
df_nondup.to_csv(DATA_OUT, index=False)
print(f"[INFO] Saved non-duplicate dataset: {DATA_OUT}")
print(f"Rows before: {len(df):,}, rows after removing duplicates: {len(df_nondup):,}")