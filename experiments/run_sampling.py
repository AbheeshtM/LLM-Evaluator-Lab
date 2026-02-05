import pandas as pd
from pathlib import Path

# -----------------------
# Config
# -----------------------
RAW_DATA_PATH = "data/raw/kaggle_dataset.csv"
OUTPUT_PATH = "data/processed/sampled_data.csv"

TOTAL_SAMPLES = 100  # Groq-safe
RANDOM_STATE = 42

# -----------------------
# Load data
# -----------------------
df = pd.read_csv(RAW_DATA_PATH)

# Clean column names
df.columns = [c.strip() for c in df.columns]

TARGET_COL = "income"
GENDER_COL = "sex"

# Clean labels
df[TARGET_COL] = df[TARGET_COL].str.strip()

# Filter valid genders
df = df[df[GENDER_COL].isin(["Male", "Female"])]

# -----------------------
# Balanced sampling
# -----------------------
samples_per_group = TOTAL_SAMPLES // 4

groups = []
for income in ["<=50K", ">50K"]:
    for gender in ["Male", "Female"]:
        subset = df[
            (df[TARGET_COL] == income) &
            (df[GENDER_COL] == gender)
        ]
        sample = subset.sample(
            n=min(samples_per_group, len(subset)),
            random_state=RANDOM_STATE
        )
        groups.append(sample)

sampled_df = (
    pd.concat(groups)
    .sample(frac=1, random_state=RANDOM_STATE)
)

# -----------------------
# Save
# -----------------------
Path("data/processed").mkdir(parents=True, exist_ok=True)
sampled_df.to_csv(OUTPUT_PATH, index=False)

print("✅ Sampling complete")
print("Final shape:", sampled_df.shape)
print(sampled_df[[TARGET_COL, GENDER_COL]].value_counts())
