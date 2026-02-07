import pandas as pd

# Load CSV
df = pd.read_csv("Final_models_with10papers_training_data.csv")

# Count rows per model_id
counts = df['label'].value_counts()

# Keep only model_ids with at least 10 rows
valid_models = counts[counts >= 10].index
filtered_df = df[df['label'].isin(valid_models)]

# Save new CSV
filtered_df.to_csv("Filtered_models_min10rows.csv", index=False)

print(f"Original rows: {len(df)}")
print(f"Filtered rows: {len(filtered_df)}")
print(f"Unique models kept: {filtered_df['label'].nunique()}")
