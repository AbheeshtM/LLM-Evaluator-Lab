import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="LLM Reliability Lab",
    layout="wide"
)

st.title("🔍 LLM Reliability Lab Dashboard")
st.caption("Evaluation of LLM accuracy, bias, and hallucinations")

# -------------------------------------------------
# Load data
# -------------------------------------------------
@st.cache_data
def load_data():
    parsed = pd.read_csv("results/parsed_outputs.csv")
    bias = pd.read_csv("results/bias_by_gender.csv", index_col=0)
    metrics = pd.read_csv("results/metrics_summary.csv")
    return parsed, bias, metrics

parsed_df, bias_df, metrics_df = load_data()

# -------------------------------------------------
# Overall metrics
# -------------------------------------------------
st.header("📌 Overall Performance")

col1, col2, col3 = st.columns(3)

with col1:
    acc = parsed_df["is_correct"].mean()
    st.metric("Accuracy", f"{acc:.2f}")

with col2:
    halluc_rate = (
        (parsed_df["confidence"] == "HIGH") &
        (parsed_df["is_correct"] == False)
    ).mean()
    st.metric("Hallucination Rate", f"{halluc_rate:.2f}")

with col3:
    st.metric("Samples Evaluated", len(parsed_df))

# -------------------------------------------------
# Accuracy vs Confidence
# -------------------------------------------------
st.header("📊 Accuracy vs Confidence")

fig1, ax1 = plt.subplots()
ax1.bar(metrics_df["confidence"], metrics_df["accuracy"])
ax1.set_ylabel("Accuracy")
ax1.set_xlabel("Confidence Level")
ax1.set_ylim(0, 1)
st.pyplot(fig1)

# -------------------------------------------------
# Bias analysis
# -------------------------------------------------
st.header("⚖️ Bias Analysis (Gender)")

fig2, ax2 = plt.subplots()
bias_df[["accuracy", "hallucination_rate"]].plot(
    kind="bar",
    ax=ax2
)
ax2.set_ylabel("Rate")
ax2.set_xlabel("Gender")
ax2.set_ylim(0, 1)
st.pyplot(fig2)

# -------------------------------------------------
# Hallucination examples
# -------------------------------------------------
st.header("🚨 Confident but Wrong Examples")

hallucinations = parsed_df[
    (parsed_df["confidence"] == "HIGH") &
    (parsed_df["is_correct"] == False)
]

st.write(
    f"Showing {min(5, len(hallucinations))} of "
    f"{len(hallucinations)} hallucination cases"
)

st.dataframe(
    hallucinations[
        ["ground_truth", "prediction", "confidence"]
    ].head(5)
)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption(
    "This dashboard visualizes precomputed LLM evaluation results. "
    "No live model inference is performed in the UI."
)
