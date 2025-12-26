
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

_USE_SRC = False
try:
    from src.io import load_dataset
    from src.ui import sidebar_controls, resample_df, calc_kpis
    from src.charts import get_figure_by_name
    from src.simulation import simulate_profit, simple_recommendation
    _USE_SRC = True
except Exception:
    pass


def _load_dataset_fallback():
    """Load dataset from project-level data/processed/final_df.*"""
    project_root = Path(__file__).parents[2]

    candidates = [
        project_root / "data/processed/final_df.parquet",
        project_root / "data/processed/final_df.csv",
        project_root / "data/final_df.parquet",
        project_root / "data/final_df.csv",
    ]

    for p in candidates:
        if p.exists():
            if p.suffix == ".parquet":
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p, parse_dates=["date"])
            df.columns = [c.lower() for c in df.columns]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df

    raise FileNotFoundError("No dataset found in /data or /data/processed")

def _sidebar_controls_fallback(df):
    st.sidebar.header("Controls")
    symbols = sorted(df["symbol"].unique())
    symbol = st.sidebar.selectbox("Symbol", symbols)

    min_date = df["date"].min().date()
    max_date = df["date"].max().date()

    start_date = st.sidebar.date_input("Start date", min_date)
    end_date = st.sidebar.date_input("End date", max_date)

    interval = st.sidebar.selectbox("Interval", ["Daily", "Weekly", "Monthly"])
    export = st.sidebar.button("Export filtered CSV")

    return {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "interval": interval,
        "export": export,
    }

def _resample_df_fallback(df, interval):
    if interval == "Daily":
        return df.copy()

    df = df.set_index("date")

    rule = "W" if interval == "Weekly" else "M"

    out = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna().reset_index()

    return out

def _calc_kpis_fallback(df_symbol):
    df2 = df_symbol.sort_values("date")
    latest = df2.iloc[-1]
    prev = df2.iloc[-2] if len(df2) >= 2 else latest
    pct = ((latest["close"] - prev["close"]) / prev["close"]) * 100 if prev["close"] else 0
    return {
        "latest_close": float(latest["close"]),
        "pct_change": float(pct),
        "volume": int(latest["volume"])
    }

import plotly.graph_objects as go
import plotly.express as px

def _candlestick(df, title):
    df = df.sort_values("date")
    fig = go.Figure(go.Candlestick(
        x=df["date"],
        open=df["open"], high=df["high"],
        low=df["low"], close=df["close"]
    ))
    fig.update_layout(title=title, xaxis_rangeslider_visible=True)
    return fig

def _line(df, title):
    fig = go.Figure(go.Scatter(x=df["date"], y=df["close"], mode="lines"))
    fig.update_layout(title=title, xaxis_rangeslider_visible=True)
    return fig

def _volume(df, title):
    fig = go.Figure(go.Bar(x=df["date"], y=df["volume"]))
    fig.update_layout(title=title, xaxis_rangeslider_visible=True)
    return fig

def _sma(df, windows=[20,50,200], title="SMA Overlay"):
    df = df.sort_values("date").copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Close"))

    for w in windows:
        df[f"sma_{w}"] = df["close"].rolling(w, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=df["date"], y=df[f"sma_{w}"], name=f"SMA {w}"))

    fig.update_layout(title=title, xaxis_rangeslider_visible=True)
    return fig

def _volatility(df, window=30, title="Rolling Volatility"):
    df = df.sort_values("date").copy()
    roll = df["close"].pct_change().rolling(window).std()
    fig = go.Figure(go.Scatter(x=df["date"], y=roll, name=f"Vol {window}"))
    fig.update_layout(title=title, xaxis_rangeslider_visible=True)
    return fig

def _drawdown(df, title="Drawdown"):
    df = df.sort_values("date").copy()
    prices = df["close"]
    running_max = prices.cummax()
    dd = (prices - running_max) / running_max

    fig = go.Figure(go.Scatter(x=df["date"], y=dd, fill="tozeroy"))
    fig.update_layout(title=title, xaxis_rangeslider_visible=True)
    return fig

def _returns_hist(df, freq="D", title="Returns Histogram"):
    df = df.sort_values("date").copy()
    if freq == "D":
        rets = df["close"].pct_change()
    else:
        temp = df.set_index("date")["close"].resample(freq).last()
        rets = temp.pct_change()
    rets = rets.dropna()

    fig = px.histogram(rets, nbins=40, title=title)
    return fig

def _recent_table(df, n=20, title="Recent Activity"):
    df = df.sort_values("date", ascending=False).head(n)
    fig = go.Figure(go.Table(
        header=dict(values=list(df.columns)),
        cells=dict(values=[df[c].tolist() for c in df.columns])
    ))
    fig.update_layout(title=title)
    return fig

def _get_figure_by_name_fallback(df, name, **kw):
    name = name.lower()
    if "candlestick" in name: return _candlestick(df, name)
    if "price" in name and "sma" not in name: return _line(df, name)
    if "volume" in name: return _volume(df, name)
    if "sma" in name: return _sma(df, **kw)
    if "volatility" in name: return _volatility(df, **kw)
    if "drawdown" in name: return _drawdown(df)
    if "hist" in name: return _returns_hist(df, **kw)
    if "recent" in name: return _recent_table(df, **kw)
    return _line(df, name)

def _simulate_profit_fallback(price, qty, sell):
    cost = price * qty
    revenue = sell * qty
    profit = revenue - cost
    pct = (profit / cost) * 100 if cost else 0
    return {"profit": profit, "profit_pct": pct}

def _simple_rec_fallback(price, expected, target=None):
    if expected > price: return "BUY"
    return "SELL"


if _USE_SRC:
    load_dataset_impl = load_dataset
    sidebar_controls_impl = sidebar_controls
    resample_df_impl = resample_df
    calc_kpis_impl = calc_kpis
    get_figure_by_name_impl = get_figure_by_name
    simulate_profit_impl = simulate_profit
    simple_recommendation_impl = simple_recommendation
else:
    load_dataset_impl = _load_dataset_fallback
    sidebar_controls_impl = _sidebar_controls_fallback
    resample_df_impl = _resample_df_fallback
    calc_kpis_impl = _calc_kpis_fallback
    get_figure_by_name_impl = _get_figure_by_name_fallback
    simulate_profit_impl = _simulate_profit_fallback
    simple_recommendation_impl = _simple_rec_fallback

@st.cache_data
def load_cached():
    return load_dataset_impl()


def dashboard_page(df):
    st.title("Crypto Dashboard — AE2")

    controls = sidebar_controls_impl(df)
    symbol = controls["symbol"]
    start = pd.to_datetime(controls["start_date"])
    end = pd.to_datetime(controls["end_date"])
    interval = controls["interval"]

    df_pair = df[df["symbol"] == symbol].sort_values("date").reset_index(drop=True)
    mask = (df_pair["date"] >= start) & (df_pair["date"] <= end)
    df_pair = df_pair[mask]

    if df_pair.empty:
        st.warning("No data for this range.")
        return

    df_pair = resample_df_impl(df_pair, interval)

    kpi = calc_kpis_impl(df_pair)
    c1, c2, c3 = st.columns(3)
    c1.metric("Price", f"£{kpi['latest_close']:.2f}")
    c2.metric("Change", f"{kpi['pct_change']:.2f}%")
    c3.metric("Volume", f"{kpi['volume']:,}")

    st.subheader("Visualisation")

    options = [
        "Price time series (line)",
        "Candlestick (OHLC)",
        "Volume",
        "SMA overlay (20,50,200)",
        "Rolling volatility",
        "Drawdown",
        "Returns histogram (daily)",
        "Recent activity (table)"
    ]

    choice = st.selectbox("Choose chart", options)

    if choice == "SMA overlay (20,50,200)":
        fig = get_figure_by_name_impl(df_pair, "sma overlay", windows=[20,50,200])
    elif choice == "Rolling volatility":
        fig = get_figure_by_name_impl(df_pair, "rolling volatility", window=30)
    elif choice == "Returns histogram (daily)":
        fig = get_figure_by_name_impl(df_pair, "returns histogram", freq="D")
    elif choice == "Recent activity (table)":
        fig = get_figure_by_name_impl(df_pair, "recent table", n=20)
    else:
        fig = get_figure_by_name_impl(df_pair, choice)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent rows")
    st.dataframe(df_pair.tail(10).set_index("date"))

    st.subheader("What-If Profit Simulator")

    colA, colB = st.columns(2)
    price = kpi["latest_close"]

    with colA:
        qty = st.number_input("Quantity", value=1.0, step=0.1)
        sell = st.number_input("Sell price", value=price * 1.05, step=0.1)
        calc = st.button("Calculate")

    with colB:
        if calc:
            res = simulate_profit_impl(price, qty, sell)
            st.metric("Profit", f"£{res['profit']:.2f}")
            st.metric("Profit %", f"{res['profit_pct']:.2f}%")
        else:
            st.info("Enter values and click Calculate.")

try:
    df_main = load_cached()
    dashboard_page(df_main)
except Exception as e:
    st.error(f"Error loading dashboard: {e}")
