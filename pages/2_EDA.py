# streamlit/pages/2_EDA.py
"""
AE2 — Exploratory Data Analysis (EDA)
Clean, CSV-driven, non-overengineered version.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="AE2 - EDA", layout="wide")

# ------------------------------------------------------------------
# Resolve project root robustly
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT.name != "crypto_forecasting_system":
    PROJECT_ROOT = PROJECT_ROOT.parent

EDA_DIR = PROJECT_ROOT / "data" / "EDA"

# ------------------------------------------------------------------
# Load summary stats (master index)
# ------------------------------------------------------------------
@st.cache_data
def load_summary():
    return pd.read_csv(EDA_DIR / "summary_stats.csv")

summary_df = load_summary()
symbols = sorted(summary_df["symbol"].unique())

# ------------------------------------------------------------------
# UI controls
# ------------------------------------------------------------------
st.title("Exploratory Data Analysis (AE2)")

col1, col2 = st.columns(2)

with col1:
    symbol = st.selectbox("Select Cryptocurrency", symbols)

with col2:
    eda_option = st.selectbox(
        "Select EDA Analysis",
        [
            "Summary Statistics",
            "Price Trend",
            "Distribution Analysis",
            "OHLCV Correlation",
            "Rolling Statistics",
            "Seasonality",
            "Outlier Detection",
            "Missing Data",
            "Volume Analysis",
            "Volatility Clustering",
            "Return Analysis",
            "Lag Features (ACF / PACF)"
        ]
    )

# ------------------------------------------------------------------
# 1. SUMMARY STATISTICS
# ------------------------------------------------------------------
if eda_option == "Summary Statistics":
    st.header("Summary Statistics")
    st.dataframe(summary_df[summary_df["symbol"] == symbol])

# ------------------------------------------------------------------
# 2. PRICE TREND
# ------------------------------------------------------------------
elif eda_option == "Price Trend":
    df = pd.read_csv(EDA_DIR / "returns" / f"{symbol}_returns.csv", parse_dates=["date"])

    fig = px.line(df, x="date", y="close", title=f"{symbol} Close Price Trend")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# 3. DISTRIBUTION ANALYSIS
# ------------------------------------------------------------------
elif eda_option == "Distribution Analysis":
    price_df = pd.read_csv(
        EDA_DIR / "distributions" / "price" / f"{symbol}_price_distribution.csv"
    )
    returns_df = pd.read_csv(
        EDA_DIR / "distributions" / "returns" / f"{symbol}_returns_distribution.csv"
    )

    st.subheader("Price Distribution")
    st.plotly_chart(px.histogram(price_df, x="close", nbins=50), use_container_width=True)

    st.subheader("Returns Distribution")
    st.plotly_chart(px.histogram(returns_df, x="returns", nbins=50), use_container_width=True)

# ------------------------------------------------------------------
# 4. OHLCV CORRELATION
# ------------------------------------------------------------------
elif eda_option == "OHLCV Correlation":
    corr = pd.read_csv(EDA_DIR / "correlation" / f"{symbol}_corr.csv", index_col=0)
    st.plotly_chart(px.imshow(corr, text_auto=True), use_container_width=True)

# ------------------------------------------------------------------
# 5. ROLLING STATISTICS
# ------------------------------------------------------------------
elif eda_option == "Rolling Statistics":
    df = pd.read_csv(EDA_DIR / "rolling" / f"{symbol}_rolling.csv", parse_dates=["date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Close"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["sma_30"], name="SMA 30"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["sma_100"], name="SMA 100"))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# 6. SEASONALITY
# ------------------------------------------------------------------
elif eda_option == "Seasonality":
    month_df = pd.read_csv(
        EDA_DIR / "seasonality" / f"{symbol}_monthly_returns.csv"
    )
    dow_df = pd.read_csv(
        EDA_DIR / "seasonality" / f"{symbol}_dow_returns.csv"
    )

    st.subheader("Monthly Returns")
    st.plotly_chart(px.bar(month_df, x="date", y="monthly_return"), use_container_width=True)

    st.subheader("Day-of-Week Returns")
    st.plotly_chart(px.bar(dow_df, x="dow", y="avg_return"), use_container_width=True)

# ------------------------------------------------------------------
# 7. OUTLIER DETECTION
# ------------------------------------------------------------------
elif eda_option == "Outlier Detection":
    out_df = pd.read_csv(
        EDA_DIR / "outliers" / f"{symbol}_outliers.csv", parse_dates=["date"]
    )

    if out_df.empty:
        st.info(
            "No outliers detected using the IQR method. "
            "This is common for large-cap cryptocurrencies like BTC."
        )
    else:
        st.dataframe(out_df)
        st.plotly_chart(
            px.scatter(out_df, x="date", y="close", title="Detected Outliers"),
            use_container_width=True
        )

# ------------------------------------------------------------------
# 8. MISSING DATA
# ------------------------------------------------------------------
elif eda_option == "Missing Data":
    miss = pd.read_csv(EDA_DIR / "missing" / "missing_summary.csv")
    st.dataframe(miss)

# ------------------------------------------------------------------
# 9. VOLUME ANALYSIS
# ------------------------------------------------------------------
elif eda_option == "Volume Analysis":
    df = pd.read_csv(
        EDA_DIR / "volume" / f"{symbol}_volume_stats.csv", parse_dates=["date"]
    )

    fig = px.line(df, x="date", y="volume", title="Volume Trend")
    fig.add_scatter(x=df["date"], y=df["volume_ma_30"], name="Volume MA 30")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# 10. VOLATILITY CLUSTERING
# ------------------------------------------------------------------
elif eda_option == "Volatility Clustering":
    df = pd.read_csv(
        EDA_DIR / "volatility" / f"{symbol}_returns_squared.csv", parse_dates=["date"]
    )

    st.plotly_chart(
        px.line(df, x="date", y="returns_squared", title="Returns Squared"),
        use_container_width=True
    )

# ------------------------------------------------------------------
# 11. RETURN ANALYSIS
# ------------------------------------------------------------------
elif eda_option == "Return Analysis":
    cum = pd.read_csv(
        EDA_DIR / "returns" / f"{symbol}_cumulative_returns.csv", parse_dates=["date"]
    )
    dd = pd.read_csv(
        EDA_DIR / "returns" / f"{symbol}_drawdown.csv", parse_dates=["date"]
    )

    st.subheader("Cumulative Returns")
    st.plotly_chart(px.line(cum, x="date", y="cumulative_returns"), use_container_width=True)

    st.subheader("Drawdown")
    st.plotly_chart(px.area(dd, x="date", y="drawdown"), use_container_width=True)

# ------------------------------------------------------------------
# 12. LAG FEATURES + ACF / PACF
# ------------------------------------------------------------------
elif eda_option == "Lag Features (ACF / PACF)":
    lag = pd.read_csv(EDA_DIR / "lag" / f"{symbol}_lags.csv")
    acf = pd.read_csv(EDA_DIR / "lag" / f"{symbol}_acf.csv")
    pacf = pd.read_csv(EDA_DIR / "lag" / f"{symbol}_pacf.csv")

    st.subheader("Lag Features")
    st.dataframe(lag.tail(15))

    st.subheader("ACF Values")
    st.dataframe(acf)

    st.subheader("PACF Values")
    st.dataframe(pacf)
