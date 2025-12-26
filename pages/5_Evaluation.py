import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ============================================================
# CONFIG
# ============================================================

EVAL_PATH = "models/evaluation_results.csv"

# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(EVAL_PATH):
    st.error("❌ evaluation_results.csv not found in models folder")
    st.stop()

df = pd.read_csv(EVAL_PATH)

# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.title("📊 Evaluation Controls")

coin_options = ["All"] + sorted(df["Symbol"].unique().tolist())
model_options = ["All"] + sorted(df["Model"].unique().tolist())

selected_coin = st.sidebar.selectbox("Select Coin", coin_options)
selected_model = st.sidebar.selectbox("Select Model", model_options)

filtered_df = df.copy()

if selected_coin != "All":
    filtered_df = filtered_df[filtered_df["Symbol"] == selected_coin]

if selected_model != "All":
    filtered_df = filtered_df[filtered_df["Model"] == selected_model]

# ============================================================
# MAIN PAGE
# ============================================================

st.title("📊 Model Evaluation & Comparison")

st.markdown(
    """
    This page evaluates forecasting models using **AE2-required regression metrics**:
    **MAE**, **RMSE**, **MAPE**, and **R²**.
    """
)

# ============================================================
# METRICS TABLE
# ============================================================

st.subheader("📄 Evaluation Metrics")

st.dataframe(
    filtered_df.sort_values(["Symbol", "RMSE"]),
    use_container_width=True
)

# ============================================================
# RMSE COMPARISON (BAR CHART)
# ============================================================

st.subheader("📉 RMSE Comparison")

rmse_fig = px.bar(
    filtered_df,
    x="Model",
    y="RMSE",
    color="Symbol",
    barmode="group",
    title="RMSE by Model and Coin"
)

st.plotly_chart(rmse_fig, use_container_width=True)

# ============================================================
# MAE COMPARISON
# ============================================================

st.subheader("📉 MAE Comparison")

mae_fig = px.bar(
    filtered_df,
    x="Model",
    y="MAE",
    color="Symbol",
    barmode="group",
    title="MAE by Model and Coin"
)

st.plotly_chart(mae_fig, use_container_width=True)

# ============================================================
# MAPE COMPARISON
# ============================================================

st.subheader("📉 MAPE (%) Comparison")

mape_fig = px.bar(
    filtered_df,
    x="Model",
    y="MAPE (%)",
    color="Symbol",
    barmode="group",
    title="MAPE (%) by Model and Coin"
)

st.plotly_chart(mape_fig, use_container_width=True)

# ============================================================
# R2 COMPARISON
# ============================================================

st.subheader("📈 R² Comparison")

r2_fig = px.bar(
    filtered_df,
    x="Model",
    y="R2",
    color="Symbol",
    barmode="group",
    title="R² by Model and Coin"
)

st.plotly_chart(r2_fig, use_container_width=True)

# ============================================================
# SUMMARY INSIGHTS (AUTO)
# ============================================================

st.subheader("🧠 Quick Insights")

best_rmse = filtered_df.loc[filtered_df["RMSE"].idxmin()]
best_r2 = filtered_df.loc[filtered_df["R2"].idxmax()]

st.markdown(
    f"""
    - **Lowest RMSE**: {best_rmse['Model']} on {best_rmse['Symbol']}
    - **Highest R²**: {best_r2['Model']} on {best_r2['Symbol']}
    """
)
