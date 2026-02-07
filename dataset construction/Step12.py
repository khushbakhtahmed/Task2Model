import pandas as pd
import re
import sys

# Read the CSV file
print(f"Reading CSV file...", flush=True)
df = pd.read_csv('output_clean_with_paper_id_min200chars.csv')
print(f"Loaded {len(df)} rows", flush=True)

# Pre-compile all unique patterns to speed up processing
print("Pre-compiling regex patterns...", flush=True)
pattern_cache = {}

def get_patterns(collapsed_label):
    """Get cached patterns for a collapsed_label"""
    if collapsed_label not in pattern_cache:
        patterns = []
        # Simple word boundary approach
        patterns.append(re.compile(r'\b' + re.escape(collapsed_label) + r'\b', re.IGNORECASE))
        
        # Individual parts for hyphenated labels
        if '-' in collapsed_label:
            for part in collapsed_label.split('-'):
                if len(part) >= 2:
                    patterns.append(re.compile(r'\b' + re.escape(part) + r'\b', re.IGNORECASE))
        
        pattern_cache[collapsed_label] = patterns
    return pattern_cache[collapsed_label]

def remove_collapsed_label(row):
    """Remove collapsed_label from input_text"""
    input_text = row['input_text']
    collapsed_label = row['collapsed_label']
    
    if not isinstance(input_text, str) or not isinstance(collapsed_label, str):
        return input_text
    
    # Get cached patterns
    patterns = get_patterns(collapsed_label)
    result = input_text
    
    # Apply all patterns
    for pattern in patterns:
        result = pattern.sub('<MODEL>', result)
    
    return result

print("Processing rows...", flush=True)
df['input_text'] = df.apply(remove_collapsed_label, axis=1)

print("Saving to CSV...", flush=True)
df.to_csv('output_clean_with_paper_id_min200chars_corrected.csv', index=False)

print("Processing complete!", flush=True)
print(f"Total rows processed: {len(df)}", flush=True)
print(f"New file saved as: output_clean_with_paper_id_min200chars_corrected.csv")

