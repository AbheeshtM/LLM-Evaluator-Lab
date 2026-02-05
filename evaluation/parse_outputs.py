import pandas as pd
import re
from evaluation.confidence_parser import parse_confidence


# -----------------------
# Load raw outputs
# -----------------------
df = pd.read_csv("results/raw_outputs.csv")

# -----------------------
# Parse LLM response
# -----------------------
def parse_response(text):
    """
    Extract prediction and confidence from LLM output.
    """
    prediction = None

    if isinstance(text, str):
        pred_match = re.search(
            r"prediction\s*[:=]\s*(<=50K|>50K)",
            text,
            re.IGNORECASE
        )
        if pred_match:
            prediction = pred_match.group(1)

    confidence = parse_confidence(text)
    return prediction, confidence

# -----------------------
# Apply parsing
# -----------------------
parsed = df["llm_response"].apply(parse_response)

df["prediction"] = parsed.apply(lambda x: x[0])
df["confidence"] = parsed.apply(lambda x: x[1])

# -----------------------
# Correctness
# -----------------------
df["is_correct"] = df["prediction"] == df["ground_truth"]

# -----------------------
# Save parsed results
# -----------------------
df.to_csv("results/parsed_outputs.csv", index=False)

print("✅ Parsing complete")
print(df[["ground_truth", "prediction", "confidence", "is_correct"]].head())
