from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ===============================
# Optional src imports
# ===============================
_USE_SRC = False
try:
    from src.io import load_dataset
    from src.ui import sidebar_controls, resample_df, calc_kpis
    from src.charts import get_figure_by_name
    from src.simulation import simulate_profit, simple_recommendation
    _USE_SRC = True
except Exception:
    pass


# ===============================
# Dataset loader (CSV ONLY)
# ===============================
def _get_csv_path() -> Path:
    """
    Resolve final_df.csv from:
    CRYPTO_FORECASTING_SYSTEM/data/processed/final_df.csv
    Works regardless of Streamlit page location.
    """
    current_file = Path(__file__).resolve()

    for parent in current_file.parents:
        data_dir = parent / "data"
        if data_dir.exists():
            csv_path = data_dir / "processed" / "final_df.csv"
            if csv_path.exists():
                return csv_path

    raise FileNotFoundError(
        "final_df.csv not found at data/processed/final_df.csv"
    )


@st.cache_data
@st.cache_data
def _load_csv_cached(csv_path: str, last_modified: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Resolve date column safely
    date_candidates = ["date", "datetime", "timestamp", "time"]
    date_col = next((c for c in date_candidates if c in df.columns), None)

    if date_col is None:
        raise ValueError(
            f"No date column found. Expected one of {date_candidates}, "
            f"found: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Standardize to 'date'
    if date_col != "date":
        df = df.rename(columns={date_col: "date"})

    return df



def load_dataset_impl():
    csv_path = _get_csv_path()
    return _load_csv_cached(
        str(csv_path),
        csv_path.stat().st_mtime  # cache invalidates when CSV changes
    )


# ===============================
# Fallback UI / logic
# ===============================
def _sidebar_controls_fallback(df):
    st.sidebar.header("Controls")

    if st.sidebar.button("🔄 Reload data"):
        st.cache_data.clear()
        st.experimental_rerun()

    symbols = sorted(df["symbol"].unique())
    symbol = st.sidebar.selectbox("Symbol", symbols)

    start_date = st.sidebar.date_input(
        "Start date", df["date"].min().date()
    )
    end_date = st.sidebar.date_input(
        "End date", df["date"].max().date()
    )

    interval = st.sidebar.selectbox("Interval", ["Daily", "Weekly", "Monthly"])

    return {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "interval": interval,
    }


def _resample_df_fallback(df, interval):
    if interval == "Daily":
        return df.copy()

    df = df.set_index("date")
    rule = "W" if interval == "Weekly" else "M"

    return (
        df.resample(rule)
        .agg(
            open="first",
            high="max",
            low="min",
            close="last",
            volume="sum",
        )
        .dropna()
        .reset_index()
    )


def _calc_kpis_fallback(df):
    df = df.sort_values("date")
    latest, prev = df.iloc[-1], df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    pct = ((latest["close"] - prev["close"]) / prev["close"]) * 100 if prev["close"] else 0
    return {
        "latest_close": float(latest["close"]),
        "pct_change": float(pct),
        "volume": int(latest["volume"]),
    }


# ===============================
# Charts (unchanged)
# ===============================
def _line(df, title):
    fig = go.Figure(go.Scatter(x=df["date"], y=df["close"], mode="lines"))
    fig.update_layout(title=title, xaxis_rangeslider_visible=True)
    return fig


# ===============================
# Bind implementations
# ===============================
if _USE_SRC:
    sidebar_controls_impl = sidebar_controls
    resample_df_impl = resample_df
    calc_kpis_impl = calc_kpis
    get_figure_by_name_impl = get_figure_by_name
    simulate_profit_impl = simulate_profit
else:
    sidebar_controls_impl = _sidebar_controls_fallback
    resample_df_impl = _resample_df_fallback
    calc_kpis_impl = _calc_kpis_fallback
    get_figure_by_name_impl = lambda df, *_a, **_k: _line(df, "Price")
    simulate_profit_impl = lambda p, q, s: {
        "profit": (s - p) * q,
        "profit_pct": ((s - p) / p) * 100 if p else 0,
    }


# ===============================
# Dashboard
# ===============================
def dashboard_page(df):
    st.title("Crypto Dashboard ")
    st.caption("Data source: final_df.csv")

    controls = sidebar_controls_impl(df)

    symbol = controls["symbol"]
    start = pd.to_datetime(controls["start_date"])
    end = pd.to_datetime(controls["end_date"])
    interval = controls["interval"]

    df_pair = df[df["symbol"] == symbol]
    df_pair = df_pair[(df_pair["date"] >= start) & (df_pair["date"] <= end)]

    if df_pair.empty:
        st.warning("No data for selected range.")
        return

    df_pair = resample_df_impl(df_pair, interval)

    kpi = calc_kpis_impl(df_pair)
    c1, c2, c3 = st.columns(3)
    c1.metric("Price", f"£{kpi['latest_close']:.2f}")
    c2.metric("Change", f"{kpi['pct_change']:.2f}%")
    c3.metric("Volume", f"{kpi['volume']:,}")

    st.plotly_chart(
        get_figure_by_name_impl(df_pair, "price"),
        use_container_width=True
    )

    st.subheader("Recent rows")
    st.dataframe(df_pair.tail(10).set_index("date"))

    st.subheader("What-If Profit Simulator")
    qty = st.number_input("Quantity", value=1.0)
    sell = st.number_input("Sell price", value=kpi["latest_close"] * 1.05)

    if st.button("Calculate"):
        res = simulate_profit_impl(kpi["latest_close"], qty, sell)
        st.metric("Profit", f"£{res['profit']:.2f}")
        st.metric("Profit %", f"{res['profit_pct']:.2f}%")


# ===============================
# Run app
# ===============================
try:
    df_main = load_dataset_impl()
    dashboard_page(df_main)
except Exception as e:
    st.error(f"Error loading dashboard: {e}")
