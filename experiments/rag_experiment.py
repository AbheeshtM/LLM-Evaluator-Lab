import pandas as pd
import re
from llm.groq_client import call_groq

print("🔹 RAG experiment started")

# -----------------------
# Load data
# -----------------------
df = pd.read_csv("data/processed/sampled_data.csv")

df.columns = (
    df.columns.str.strip()
      .str.replace(".", "_")
      .str.replace("-", "_")
)

with open("prompts/decision_prompt.txt") as f:
    BASE_PROMPT = f.read()

# -----------------------
# Simple retrieval: same education + occupation
# -----------------------
def retrieve_context(row, k=3):
    similar = df[
        (df["education"] == row["education"]) &
        (df["occupation"] == row["occupation"])
    ].head(k)

    context = []
    for _, r in similar.iterrows():
        context.append(
            f"- Age {r['age']}, Hours {r['hours_per_week']}, Income {r['income']}"
        )

    return "\n".join(context)

def extract_prediction(text):
    if isinstance(text, str):
        m = re.search(r"Prediction:\s*(<=50K|>50K)", text)
        if m:
            return m.group(1)
    return None

# -----------------------
# Run comparison
# -----------------------
TESTS = 10
no_rag_correct = 0
rag_correct = 0

for i in range(TESTS):
    row = df.iloc[i]

    print(f"Testing sample {i+1}/{TESTS}")

    # ---------- No RAG ----------
    prompt_no_rag = BASE_PROMPT.format(
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

    resp_no_rag = call_groq(prompt_no_rag)
    pred_no_rag = extract_prediction(resp_no_rag)

    if pred_no_rag == row["income"]:
        no_rag_correct += 1

    # ---------- With RAG ----------
    context = retrieve_context(row)

    prompt_rag = f"""
You are given historical examples below.
Use them ONLY as supporting evidence.

Examples:
{context}

{prompt_no_rag}
"""

    resp_rag = call_groq(prompt_rag)
    pred_rag = extract_prediction(resp_rag)

    if pred_rag == row["income"]:
        rag_correct += 1

# -----------------------
# Results
# -----------------------
print("\n=== RAG vs No-RAG ===")
print(f"No-RAG accuracy: {no_rag_correct}/{TESTS}")
print(f"RAG accuracy: {rag_correct}/{TESTS}")

print("✅ RAG experiment completed")
