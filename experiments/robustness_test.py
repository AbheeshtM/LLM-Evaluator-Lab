import pandas as pd
import re
from llm.groq_client import call_groq

print("🔹 Robustness test started")

# -----------------------
# Load data
# -----------------------
df = pd.read_csv("data/processed/sampled_data.csv")

# Normalize columns
df.columns = (
    df.columns.str.strip()
      .str.replace(".", "_")
      .str.replace("-", "_")
)

with open("prompts/decision_prompt.txt") as f:
    PROMPT = f.read()

# -----------------------
# Helper to extract prediction
# -----------------------
def extract_prediction(text):
    if isinstance(text, str):
        match = re.search(r"Prediction:\s*(<=50K|>50K)", text)
        if match:
            return match.group(1)
    return None

# -----------------------
# Robustness experiment
# -----------------------
TOTAL_TESTS = 10  # keep SMALL (important)
decision_flips = 0

for i in range(TOTAL_TESTS):
    row = df.iloc[i].copy()

    print(f"Testing sample {i+1}/{TOTAL_TESTS}")

    # Original prompt
    prompt_original = PROMPT.format(
        age=row["age"],
        workclass=row["workclass"],
        education=row["education"],
        marital_status=row["marital_status"],
        occupation=row["occupation"],
        relationship=row["relationship"],
        race=row["race"],
        sex=row["sex"],
        hours_per_week=row["hours_per_week"],
    )

    # Perturbation (small, realistic)
    row["hours_per_week"] += 5

    prompt_perturbed = PROMPT.format(
        age=row["age"],
        workclass=row["workclass"],
        education=row["education"],
        marital_status=row["marital_status"],
        occupation=row["occupation"],
        relationship=row["relationship"],
        race=row["race"],
        sex=row["sex"],
        hours_per_week=row["hours_per_week"],
    )

    # LLM calls (slow by design)
    resp_orig = call_groq(prompt_original)
    resp_pert = call_groq(prompt_perturbed)

    pred_orig = extract_prediction(resp_orig)
    pred_pert = extract_prediction(resp_pert)

    if pred_orig != pred_pert:
        decision_flips += 1
        print("⚠️ Decision flip detected")

print("\n=== Robustness Results ===")
print(f"Decision flips: {decision_flips}/{TOTAL_TESTS}")
print(f"Flip rate: {round(decision_flips / TOTAL_TESTS, 2)}")

print("✅ Robustness test completed")
