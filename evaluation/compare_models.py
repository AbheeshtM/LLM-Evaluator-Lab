import pandas as pd

MODELS = {
    "LLaMA-3.1-8B": "results/llama_3_1_8b",
    "Qwen-32B": "results/qwen_32b",
}

rows = []

for model, path in MODELS.items():
    parsed = pd.read_csv(f"{path}/parsed_outputs.csv")
    bias = pd.read_csv(f"{path}/bias_by_gender.csv")

    accuracy = parsed["is_correct"].mean()

    halluc_rate = (
        (parsed["confidence"] == "HIGH") &
        (parsed["is_correct"] == False)
    ).mean()

    rows.append({
        "model": model,
        "accuracy": round(accuracy, 3),
        "hallucination_rate": round(halluc_rate, 3),
        "female_accuracy": round(bias.loc["Female", "accuracy"], 3),
        "male_accuracy": round(bias.loc["Male", "accuracy"], 3),
    })

compare_df = pd.DataFrame(rows)
print("\n=== Model Comparison ===")
print(compare_df)
