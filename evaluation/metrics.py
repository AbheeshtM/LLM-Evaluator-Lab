import pandas as pd

df = pd.read_csv("results/parsed_outputs.csv")

# Overall accuracy
overall_accuracy = df["is_correct"].mean()

# Accuracy by confidence level
confidence_metrics = (
    df.groupby("confidence")
      .agg(
          total=("confidence", "count"),
          accuracy=("is_correct", "mean")
      )
)

print("\n=== Overall Accuracy ===")
print(round(overall_accuracy, 3))

print("\n=== Accuracy by Confidence ===")
print(confidence_metrics)

# Save metrics
confidence_metrics.to_csv("results/metrics_summary.csv")
print("\nSaved -> results/metrics_summary.csv")
