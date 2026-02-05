import subprocess
import sys

print("🚀 Starting full LLM evaluation pipeline\n")

STEPS = [
    ("Sampling data", ["python", "-m", "experiments.run_sampling"]),
    ("Baseline LLM inference", ["python", "-m", "experiments.run_baseline"]),
    ("Parsing outputs", ["python", "-m", "evaluation.parse_outputs"]),
    ("Bias analysis", ["python", "-m", "experiments.bias_analysis"]),
    ("Metrics calculation", ["python", "-m", "evaluation.metrics"]),
    # Optional:
    # ("Robustness test", ["python", "-m", "experiments.robustness_test"]),
]

for name, command in STEPS:
    print(f"\n▶️ {name}")
    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\n❌ Failed at step: {name}")
        sys.exit(1)

print("\n✅ Pipeline completed successfully")
