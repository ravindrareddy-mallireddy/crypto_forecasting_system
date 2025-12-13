"""
AE2 — Clustering Analysis Page
Uses precomputed clustering outputs from Colab.
"""

import json
import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

st.set_page_config(page_title="AE2 - Clustering", layout="wide")

# -------------------------------------------------
# Resolve project root robustly
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT.name != "crypto_forecasting_system":
    PROJECT_ROOT = PROJECT_ROOT.parent

EDA_DIR = PROJECT_ROOT / "data" / "EDA" / "clustering"

# -------------------------------------------------
# Load clustering outputs
# -------------------------------------------------
@st.cache_data
def load_cluster_labels():
    return pd.read_csv(EDA_DIR / "cluster_labels.csv")

@st.cache_data
def load_cluster_groups():
    with open(EDA_DIR / "cluster_groups.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_representatives():
    return pd.read_csv(EDA_DIR / "cluster_representatives.csv")

@st.cache_data
def load_representative_metrics():
    return pd.read_csv(EDA_DIR / "cluster_representatives_reasoning.csv")

def load_cluster_correlation(coin: str):
    df = pd.read_csv(EDA_DIR / f"cluster_correlation_{coin}.csv")
    df.columns = [c.lower() for c in df.columns]  # normalize
    return df


clusters_df = load_cluster_labels()
cluster_groups = load_cluster_groups()
rep_df = load_representatives()
rep_metrics_df = load_representative_metrics()

# -------------------------------------------------
# Page Header
# -------------------------------------------------
st.title("Cryptocurrency Clustering Analysis (AE2)")
st.markdown(
    """
This page presents **unsupervised clustering of cryptocurrencies**
using engineered statistical features and **PCA-based dimensionality reduction**.
The analysis identifies groups of cryptocurrencies that exhibit similar
market behaviour.
"""
)

# -------------------------------------------------
# 1. PCA Cluster Visualisation
# -------------------------------------------------
st.header("Cluster Visualisation (PCA Projection)")

fig = px.scatter(
    clusters_df,
    x="pca_1",
    y="pca_2",
    color="cluster",
    hover_data=["symbol"],
    title="PCA Scatter Plot of Cryptocurrency Clusters"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# 2. Cluster Composition & Interpretation
# -------------------------------------------------
st.header("Cluster Composition & Interpretation")

for cluster_id in sorted(cluster_groups.keys(), key=int):
    coins = cluster_groups[str(cluster_id)]

    representative = rep_df.loc[
        rep_df["cluster"] == int(cluster_id), "representative_coin"
    ].values[0]

    metrics_row = rep_metrics_df.loc[
        rep_metrics_df["cluster"] == int(cluster_id)
    ].iloc[0]

    avg_vol = metrics_row["avg_volume"]
    obs = int(metrics_row["obs_count"])
    avg_dd = metrics_row["avg_drawdown"]

    reason_text = (
        f"The representative coin was selected based on its high average trading volume "
        f"({avg_vol:,.0f}), sufficient historical observations ({obs} days), "
        f"and characteristic drawdown behaviour (average drawdown ≈ {avg_dd:.2%}). "
        f"This makes it a robust and informative representative of this cluster."
    )

    with st.expander(f"Cluster {cluster_id}"):
        st.markdown(f"**Number of coins:** {len(coins)}")
        st.markdown("**Coins in this cluster:**")
        st.write(", ".join(coins))

        st.markdown(f"**Representative coin:** `{representative}`")
        st.markdown("**Reason for selection:**")
        st.info(reason_text)

# -------------------------------------------------
# 3. Correlation Analysis for Representative Coins
# -------------------------------------------------
st.header("Correlation Analysis for Representative Coins")

st.markdown(
    """
For each cluster’s representative coin, the **top 4 positively**
and **top 4 negatively correlated cryptocurrencies** are shown.
Correlations are computed using **daily returns**.
"""
)

for _, row in rep_df.iterrows():
    cluster_id = row["cluster"]
    coin = row["representative_coin"]

    corr_df = load_cluster_correlation(coin)

    top_pos = (
        corr_df[corr_df["correlation"] > 0]
        .sort_values("correlation", ascending=False)
        .head(4)
    )

    top_neg = (
        corr_df[corr_df["correlation"] < 0]
        .sort_values("correlation")
        .head(4)
    )

    st.subheader(f"Cluster {cluster_id} — {coin}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top 4 Positively Correlated Coins**")
        st.dataframe(
            top_pos
            .assign(Rank=range(1, len(top_pos) + 1))
            .set_index("Rank")
            .style.format({"correlation": "{:.3f}"})
        )

    with col2:
        st.markdown("**Top 4 Negatively Correlated Coins**")
        st.dataframe(
            top_neg
            .assign(Rank=range(1, len(top_neg) + 1))
            .set_index("Rank")
            .style.format({"Correlation": "{:.3f}"})
        )

st.caption(
    "Correlation values are based on daily percentage returns. "
    "Positive values indicate similar price movement, while "
    "negative values indicate inverse behaviour."
)
