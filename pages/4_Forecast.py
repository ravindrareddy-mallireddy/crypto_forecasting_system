"""
AE2 — Forecasting Page
Reads precomputed forecasting outputs (NO model training here).
"""

import json
import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

st.set_page_config(page_title="AE2 - Forecasting", layout="wide")

# -------------------------------------------------
# Resolve project root robustly
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT.name != "crypto_forecasting_system":
    PROJECT_ROOT = PROJECT_ROOT.parent

FORECAST_DIR = PROJECT_ROOT / "data" / "forecasting"

# -------------------------------------------------
# Helpers
# -------------------------------------------------
@st.cache_data
def list_coins():
    return sorted([p.name for p in FORECAST_DIR.iterdir() if p.is_dir()])

@st.cache_data
def load_metrics(coin):
    return pd.read_csv(FORECAST_DIR / coin / "metrics.csv")

@st.cache_data
def load_best_model(coin):
    with open(FORECAST_DIR / coin / "best_model.json") as f:
        return json.load(f)

@st.cache_data
def load_model_notes(coin):
    with open(FORECAST_DIR / coin / "model_notes.json") as f:
        return json.load(f)

@st.cache_data
def load_actual_vs_predicted(coin, model):
    return pd.read_csv(
        FORECAST_DIR / coin / f"actual_vs_predicted_{model}.csv",
        parse_dates=["date"]
    )

@st.cache_data
def load_future_forecast(coin, model):
    return pd.read_csv(
        FORECAST_DIR / coin / f"future_forecast_{model}.csv",
        parse_dates=["date"]
    )

# -------------------------------------------------
# UI Controls
# -------------------------------------------------
st.title("Cryptocurrency Forecasting (AE2)")

coins = list_coins()

col1, col2, col3 = st.columns(3)

with col1:
    coin = st.selectbox("Select Cryptocurrency", coins)

metrics_df = load_metrics(coin)
models = metrics_df["model"].tolist()

with col2:
    model = st.selectbox("Select Forecasting Model", models)

with col3:
    horizon_label = st.selectbox(
        "Forecast Horizon",
        {
            "1 Day": 1,
            "7 Days": 7,
            "1 Month": 30,
            "6 Months": 180,
            "1 Year": 365,
        }.items(),
        format_func=lambda x: x[0],
    )

horizon_days = horizon_label[1]

# -------------------------------------------------
# Load Data
# -------------------------------------------------
avp_df = load_actual_vs_predicted(coin, model)
future_df = load_future_forecast(coin, model)
future_df = future_df[future_df["horizon_days"] == horizon_days]

best_model_info = load_best_model(coin)
best_model = best_model_info["best_model"]

model_notes = load_model_notes(coin)

# -------------------------------------------------
# Date range selector (historical)
# -------------------------------------------------
min_date = avp_df["date"].min().date()
max_date = avp_df["date"].max().date()

date_range = st.date_input(
    "Select Historical Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date,
)

avp_df = avp_df[
    (avp_df["date"].dt.date >= date_range[0]) &
    (avp_df["date"].dt.date <= date_range[1])
]

# -------------------------------------------------
# Forecast Plot
# -------------------------------------------------
st.header("Actual vs Predicted & Future Forecast")

fig = go.Figure()

# Actual
fig.add_trace(
    go.Scatter(
        x=avp_df["date"],
        y=avp_df["actual_close"],
        name="Actual Price",
        line=dict(color="black")
    )
)

# Predicted
fig.add_trace(
    go.Scatter(
        x=avp_df["date"],
        y=avp_df["predicted_close"],
        name=f"Predicted ({model})",
        line=dict(dash="dash")
    )
)

# Future forecast
fig.add_trace(
    go.Scatter(
        x=future_df["date"],
        y=future_df["forecast_close"],
        name="Forecast",
        line=dict(color="blue")
    )
)

# Confidence interval
fig.add_trace(
    go.Scatter(
        x=list(future_df["date"]) + list(future_df["date"][::-1]),
        y=list(future_df["upper_ci"]) + list(future_df["lower_ci"][::-1]),
        fill="toself",
        fillcolor="rgba(0,0,255,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="Confidence Interval"
    )
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price",
    legend_title="Series",
    margin=dict(t=40, b=10),
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# Model Evaluation
# -------------------------------------------------
st.header("Model Evaluation")

col1, col2 = st.columns([2, 1])

with col1:
    st.dataframe(
        metrics_df
        .sort_values("RMSE")
        .style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE": "{:.2f}%"})
    )

with col2:
    st.metric(
        "Best Model (RMSE)",
        best_model.upper()
    )

    if model == best_model:
        st.success("Selected model is the best-performing model for this coin.")
    else:
        st.warning(
            f"Selected model is not the best. "
            f"Best model for {coin} is **{best_model.upper()}**."
        )

# -------------------------------------------------
# Model Notes
# -------------------------------------------------
st.header("Model Explanation")

st.info(model_notes.get(model, "No notes available for this model."))

st.caption(
    "Forecasts are generated using historical data only. "
    "Confidence intervals represent statistical uncertainty and "
    "do not guarantee future performance."
)
