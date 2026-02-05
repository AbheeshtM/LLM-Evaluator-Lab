import pandas as pd

print("🔹 Bias analysis started")

# -----------------------
# Load parsed outputs
# -----------------------
df = pd.read_csv("results/parsed_outputs.csv")
print("Loaded parsed_outputs.csv:", df.shape)
print("Columns:", df.columns.tolist())

# -----------------------
# Recompute hallucination
# -----------------------
df["is_hallucination"] = (
    (df["is_correct"] == False) &
    (df["confidence"] == "HIGH")
)
print("Hallucination column created")

# -----------------------
# Load sampled data
# -----------------------
meta = pd.read_csv("data/processed/sampled_data.csv")
print("Loaded sampled_data.csv:", meta.shape)

# Normalize column names
meta.columns = (
    meta.columns
        .str.strip()
        .str.replace(".", "_")
        .str.replace("-", "_")
)

print("Meta columns:", meta.columns.tolist())

# -----------------------
# Merge gender
# -----------------------
df = df.merge(
    meta[["sex"]],
    left_index=True,
    right_index=True
)

print("Merge complete. Shape:", df.shape)

# -----------------------
# Bias metrics
# -----------------------
bias_stats = (
    df.groupby("sex")
      .agg(
          total_samples=("sex", "count"),
          accuracy=("is_correct", "mean"),
          error_rate=("is_correct", lambda x: 1 - x.mean()),
          hallucination_rate=("is_hallucination", "mean")
      )
)

print("\n=== Bias Analysis by Gender ===")
print(bias_stats)

bias_stats.to_csv("results/bias_by_gender.csv")
print("\nSaved -> results/bias_by_gender.csv")
