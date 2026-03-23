#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re

# ============================================================
# 0. Load Data
# ============================================================
RAW_PATH = "data/ml_challenge_dataset.csv"
OUTPUT_PATH = "data/cleaned_data.csv"

df = pd.read_csv(RAW_PATH)
print(f"origin data: {df.shape[0]} row × {df.shape[1]} col")
print(f"Total missing values: {df.isnull().sum().sum()}")

# ============================================================
# 1. Rename Columns
# ============================================================
column_rename = {
    df.columns[0]:  'unique_id',
    df.columns[1]:  'painting',
    df.columns[2]:  'emotion_intensity',
    df.columns[3]:  'feeling_desc',
    df.columns[4]:  'sombre',
    df.columns[5]:  'content',
    df.columns[6]:  'calm',
    df.columns[7]:  'uneasy',
    df.columns[8]:  'num_colours',
    df.columns[9]:  'num_objects',
    df.columns[10]: 'willing_to_pay',
    df.columns[11]: 'room',
    df.columns[12]: 'view_with',
    df.columns[13]: 'season',
    df.columns[14]: 'food',
    df.columns[15]: 'soundtrack',
}
df.rename(columns=column_rename, inplace=True)
print("\n✓ Complete Column Rename")

# ===========================================================
# 2. Delete Useless Column unique_id
# ============================================================
df.drop(columns=['unique_id'], inplace=True)
print("✓ deleted unique_id column")

# ============================================================
# 3. Target variable encoding (painting → Integer Labels)
# ============================================================
target_map = {
    'The Persistence of Memory': 0,
    'The Starry Night': 1,
    'The Water Lily Pond': 2,
}
df['target'] = df['painting'].map(target_map)
print(f"✓ Target variable encoding: {target_map}")
print(f"distribution:\n{df['target'].value_counts().sort_index().to_string()}")

# ============================================================
# 4. Likert scale column → Values 1–5
#    Original Format: "1 - Strongly disagree" ... "5 - Strongly agree"
# ============================================================
LIKERT_MAP = {
    '1 - Strongly disagree': 1,
    '2 - Disagree': 2,
    '3 - Neutral/Unsure': 3,
    '4 - Agree': 4,
    '5 - Strongly agree': 5,
}
likert_cols = ['sombre', 'content', 'calm', 'uneasy']

for col in likert_cols:
    original_missing = df[col].isnull().sum()
    df[col] = df[col].map(LIKERT_MAP)         # Mapped to numerical values
    new_missing = df[col].isnull().sum()
    # After mapping, only the originally missing values will be NaN (there is no case where the mapping fails
    # because there are only 5 possible values).
    print(f"✓ Likert [{col}]: Missing {original_missing} → NaN, Mapped type {df[col].dtype}")

# ============================================================
# 5. emotion_intensity cleaning
#    Originally a float, its range should be [1, 10], but it contains 0 and decimals (6.7, 7.5, etc.).
#    Strategy: Retain [0, 10], do not process decimals (valid input), 0 is considered a valid answer.
# ============================================================
col = 'emotion_intensity'
before = df[col].isnull().sum()
print(f"\n✓ emotion_intensity: Missing {before}, Range [{df[col].min()}, {df[col].max()}]")

# ============================================================
# 6. num_colours / num_objects Outlier clipping
#    Problem: Extreme values exist (e.g., 100, 43), which is clearly random input.
#    Strategy: Clip to [0, 20] — It's impossible for a painting to have more than 20 colors/objects.
# ============================================================
for col in ['num_colours', 'num_objects']:
    n_outlier = (df[col] > 20).sum()
    df[col] = df[col].clip(lower=0, upper=20)
    print(f"✓ {col}: Clip {n_outlier} outliers (>20 → 20)")

# ============================================================
# 7. willing_to_pay cleaning — This is the dirtiest column
#
#    Original content is extremely messy:
#      Pure numbers:    "0", "150", "90000"
#      With symbols:    "$5", "200$", "$1,000", "5 000 000$"
#      With words:    "300 dollars.", "100 bucks", "200 CAD"
#      With sentences:    "I would pay about $1000 for this painting."
#      Pure text:    "a", "Not sure", "pancakes,"
#      Huge numbers:    "100000000", "100 million"
#
#    Cleaning strategy:
#      a) Remove $, commas, spaces
#      b) Handle "million" / "billion" text multipliers
#      c) Extract first number using regex
#      d) Completely unextractable → NaN
#      e) Outlier clipping: > 10,000,000 considered as abnormal → NaN
# ============================================================
def clean_willing_to_pay(val):
    """From messy willing_to_pay text, extract numeric value (CAD)"""
    if pd.isna(val):
        return np.nan
    text = str(val).lower().strip()

    # Handle "million" / "billion" text multipliers
    multiplier = 1
    if 'billion' in text:
        multiplier = 1_000_000_000
    elif 'million' in text:
        multiplier = 1_000_000

    # Remove common non-numeric characters: $, commas, CAD, dollars, bucks, etc.
    text = text.replace('$', '').replace(',', '').replace('cad', '').replace('dollars', '')
    text = text.replace('dollar', '').replace('bucks', '').replace('buck', '')

    # Processing space-separated numbers, like "5 000 000" → "5000000", "100 000" → "100000"
    # Matches only consecutive "number + space + number" patterns.
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)

    # Extract first number (integer or decimal)
    match = re.search(r'(\d+\.?\d*)', text)
    if not match:
        return np.nan

    value = float(match.group(1))

    # If text has million/billion but extracted number is small, apply multiplier
    if multiplier > 1 and value < 1000:
        value *= multiplier

    # Extreme value filtering: values over 10000000 considered as abnormal
    if value > 10_000_000:
        return np.nan

    return value

df['willing_to_pay'] = df['willing_to_pay'].apply(clean_willing_to_pay)

valid_count = df['willing_to_pay'].notna().sum()
nan_count = df['willing_to_pay'].isna().sum()
print(f"\n✓ willing_to_pay cleaning completed:")
print(f"  Valid values: {valid_count}, Unparseable/abnormal: {nan_count}")
print(f"  Range: [{df['willing_to_pay'].min()}, {df['willing_to_pay'].max()}]")
print(f"  Median: {df['willing_to_pay'].median()}")

# ===========================================================
# 8. season — Multiple selection columns, comma-separated → Multi-label binarization
#    Possible values: Spring, Summer, Fall, Winter (and combinations)
#    Example: "Spring,Summer" → season_Spring=1, season_Summer=1, others=0
# ============================================================
SEASONS = ['Spring', 'Summer', 'Fall', 'Winter']
for s in SEASONS:
    df[f'season_{s}'] = df['season'].fillna('').str.contains(s, case=False).astype(int)

print(f"\n✓ season multi-label binarization → {len(SEASONS)} columns: {['season_' + s for s in SEASONS]}")
# Verification
print(f"  Season selection frequency:")
for s in SEASONS:
    print(f"    season_{s}: {df[f'season_{s}'].sum()} ({df[f'season_{s}'].mean()*100:.1f}%)")

# ============================================================
# 9. room — Multi-select column → Multi-label binarization
#    Possible values: Bedroom, Bathroom, Office, Living room, Dining room
# ============================================================
ROOMS = ['Bedroom', 'Bathroom', 'Office', 'Living room', 'Dining room']
for r in ROOMS:
    col_name = 'room_' + r.replace(' ', '_')
    df[col_name] = df['room'].fillna('').str.contains(r, case=False).astype(int)

room_cols = ['room_' + r.replace(' ', '_') for r in ROOMS]
print(f"\n✓ room multi-label binarization → {len(ROOMS)} columns: {room_cols}")
for rc in room_cols:
    print(f"    {rc}: {df[rc].sum()} ({df[rc].mean()*100:.1f}%)")

# ============================================================
# 10. view_with — Multi-select column → Multi-label binarization
#     Possible values: Friends, Family members, Coworkers/Classmates,
#                 Strangers, By yourself
# ============================================================
VIEWERS = ['Friends', 'Family members', 'Coworkers/Classmates', 'Strangers', 'By yourself']
for v in VIEWERS:
    col_name = 'view_' + v.replace(' ', '_').replace('/', '_')
    df[col_name] = df['view_with'].fillna('').str.contains(re.escape(v), case=False).astype(int)

view_cols = ['view_' + v.replace(' ', '_').replace('/', '_') for v in VIEWERS]
print(f"\n✓ view_with multi-label binarization → {len(VIEWERS)} columns: {view_cols}")
for vc in view_cols:
    print(f"    {vc}: {df[vc].sum()} ({df[vc].mean()*100:.1f}%)")

# ============================================================
# 11. Fill missing values — Fill numeric columns with the median
#     Fill targets: emotion_intensity, num_colours, num_objects,
#              willing_to_pay, sombre, content, calm, uneasy
# ============================================================
print(f"\n--- Missing value filling (median) ---")
fill_cols = ['emotion_intensity', 'sombre', 'content', 'calm', 'uneasy',
             'num_colours', 'num_objects', 'willing_to_pay']
for col in fill_cols:
    n_missing = df[col].isnull().sum()
    if n_missing > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  ✓ {col}: filled {n_missing} missing values, median={median_val}")
    else:
        print(f"  ✓ {col}: no missing values")

# ============================================================
# 12. Free text column missing values → Empty string
#     feeling_desc, food, soundtrack left for subsequent TF-IDF
# ============================================================
text_cols = ['feeling_desc', 'food', 'soundtrack']
for col in text_cols:
    n_missing = df[col].isnull().sum()
    df[col] = df[col].fillna('')
    print(f"  ✓ {col}: {n_missing} missing values → empty string")

# ============================================================
# 13. Delete the original multi-select column that has been split up (information transferred to binary columns)
# ============================================================
df.drop(columns=['painting', 'season', 'room', 'view_with'], inplace=True)
print(f"\n✓ Deleted original multi-select columns: painting, season, room, view_with")

# ============================================================
# 14. Sort column order, output
# ============================================================
# Final columns: target + numerical features + binary features + text column
num_feature_cols = ['emotion_intensity', 'sombre', 'content', 'calm', 'uneasy',
                    'num_colours', 'num_objects', 'willing_to_pay']
binary_feature_cols = ([f'season_{s}' for s in SEASONS]
                     + room_cols
                     + view_cols)
final_order = ['target'] + num_feature_cols + binary_feature_cols + text_cols
df = df[final_order]

# ============================================================
# 15. Final check
# ============================================================
print(f"\n{'='*60}")
print(f"Cleaning complete!")
print(f"{'='*60}")
print(f"  Output size: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Numerical features: {len(num_feature_cols)} columns")
print(f"  Binary features: {len(binary_feature_cols)} columns")
print(f"  Text features: {len(text_cols)} columns (TF-IDF pending)")
print(f"  Remaining missing values: {df[num_feature_cols + binary_feature_cols].isnull().sum().sum()}")
print(f"\nColumn overview:")
for i, col in enumerate(df.columns):
    dtype_str = str(df[col].dtype)
    sample = df[col].iloc[5]
    print(f"  {i:2d}. {col:30s} | {dtype_str:8s} | Example: {sample}")

# Save
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✓ Saved to {OUTPUT_PATH}")