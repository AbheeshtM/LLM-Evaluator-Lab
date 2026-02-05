import pandas as pd

# -----------------------
# Load parsed outputs
# -----------------------
df = pd.read_csv("results/parsed_outputs.csv")

# -----------------------
# Define hallucinations
# -----------------------
df["is_hallucination"] = (
    (df["is_correct"] == False) &
    (df["confidence"] == "HIGH")
)

# -----------------------
# Metrics
# -----------------------
total = len(df)
incorrect = (~df["is_correct"]).sum()
hallucinations = df["is_hallucination"].sum()

print("Total samples:", total)
print("Incorrect predictions:", incorrect)
print("Hallucinations (confident but wrong):", hallucinations)

if incorrect > 0:
    print("Hallucination rate among errors:",
          round(hallucinations / incorrect, 3))

# -----------------------
# Save hallucinations
# -----------------------
hallucinated_df = df[df["is_hallucination"]]
hallucinated_df.to_csv("results/hallucinations.csv", index=False)

print("Saved -> results/hallucinations.csv")
