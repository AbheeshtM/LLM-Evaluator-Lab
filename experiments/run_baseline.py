import pandas as pd
from llm.groq_client import call_groq

# -----------------------
# Load sampled data
# -----------------------
df = pd.read_csv("data/processed/sampled_data.csv")

# 🔑 Normalize column names ONCE
df.columns = (
    df.columns
      .str.strip()
      .str.replace(".", "_")
      .str.replace("-", "_")
)

# -----------------------
# Load prompt template
# -----------------------
with open("prompts/decision_prompt.txt", "r", encoding="utf-8") as f:
    PROMPT_TEMPLATE = f.read()

outputs = []

# -----------------------
# Run baseline LLM evaluation
# -----------------------
for idx, row in df.iterrows():
    prompt = PROMPT_TEMPLATE.format(
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

    print(f"Running row {idx + 1}/{len(df)}")

    response = call_groq(prompt)

    outputs.append({
        "row_id": idx,
        "ground_truth": row["income"],
        "llm_response": response
    })

# -----------------------
# Save raw outputs
# -----------------------
out_df = pd.DataFrame(outputs)
out_df.to_csv("results/raw_outputs.csv", index=False)

print("✅ Baseline evaluation complete")
print("Saved to results/raw_outputs.csv")
