import streamlit as st
import pandas as pd
import plotly.express as px
import os


EVAL_PATH = "models/evaluation_results.csv"


if not os.path.exists(EVAL_PATH):
    st.error("evaluation_results.csv not found in models folder")
    st.stop()

df = pd.read_csv(EVAL_PATH)


st.sidebar.title(" Evaluation Controls")

coin_options = ["All"] + sorted(df["Symbol"].unique().tolist())
model_options = ["All"] + sorted(df["Model"].unique().tolist())

selected_coin = st.sidebar.selectbox("Select Coin", coin_options)
selected_model = st.sidebar.selectbox("Select Model", model_options)

filtered_df = df.copy()

if selected_coin != "All":
    filtered_df = filtered_df[filtered_df["Symbol"] == selected_coin]

if selected_model != "All":
    filtered_df = filtered_df[filtered_df["Model"] == selected_model]


st.title("Model Evaluation & Comparison")

st.markdown(
    """
    This page evaluates forecasting models using **AE2-required regression metrics**:
    **MAE**, **RMSE**, **MAPE**, and **R²**.
    """
)


st.subheader("Evaluation Metrics")

st.dataframe(
    filtered_df.sort_values(["Symbol", "RMSE"]),
    use_container_width=True
)


st.subheader("RMSE Comparison")

rmse_fig = px.bar(
    filtered_df,
    x="Model",
    y="RMSE",
    color="Symbol",
    barmode="group",
    title="RMSE by Model and Coin"
)

st.plotly_chart(rmse_fig, use_container_width=True)


st.subheader("MAE Comparison")

mae_fig = px.bar(
    filtered_df,
    x="Model",
    y="MAE",
    color="Symbol",
    barmode="group",
    title="MAE by Model and Coin"
)

st.plotly_chart(mae_fig, use_container_width=True)


st.subheader("MAPE (%) Comparison")

mape_fig = px.bar(
    filtered_df,
    x="Model",
    y="MAPE (%)",
    color="Symbol",
    barmode="group",
    title="MAPE (%) by Model and Coin"
)

st.plotly_chart(mape_fig, use_container_width=True)


st.subheader("R² Comparison")

r2_fig = px.bar(
    filtered_df,
    x="Model",
    y="R2",
    color="Symbol",
    barmode="group",
    title="R² by Model and Coin"
)

st.plotly_chart(r2_fig, use_container_width=True)


st.subheader("Quick Insights")

best_rmse = filtered_df.loc[filtered_df["RMSE"].idxmin()]
best_r2 = filtered_df.loc[filtered_df["R2"].idxmax()]

st.markdown(
    f"""
    - **Lowest RMSE**: {best_rmse['Model']} on {best_rmse['Symbol']}
    - **Highest R²**: {best_r2['Model']} on {best_r2['Symbol']}
    """
)
