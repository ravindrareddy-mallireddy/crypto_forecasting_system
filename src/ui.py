# src/ui.py
"""
UI controls and simple dataframe aggregation helpers for Streamlit frontend.

Functions
---------
- sidebar_controls(df) -> dict
- resample_df(df, interval) -> pd.DataFrame
- calc_kpis(df_symbol) -> dict
"""

import streamlit as st
import pandas as pd

def sidebar_controls(df: pd.DataFrame) -> dict:
    """
    Render sidebar controls and return a dictionary with:
      {'symbol', 'start_date', 'end_date', 'interval', 'export'}
    """
    st.sidebar.header("Controls")
    symbols = sorted(df["symbol"].dropna().unique().tolist())
    symbol = st.sidebar.selectbox("Select coin / symbol", symbols, index=0 if symbols else None)
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    start_date = st.sidebar.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End date", max_date, min_value=min_date, max_value=max_date)
    interval = st.sidebar.selectbox("Interval", ["Daily", "Weekly", "Monthly"])
    export = st.sidebar.button("Export filtered CSV")
    if start_date > end_date:
        st.sidebar.error("Start date must be <= End date.")
    return {"symbol": symbol, "start_date": start_date, "end_date": end_date, "interval": interval, "export": export}

def resample_df(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Resample time-series df (expects 'date' indexable) to Daily/Weekly/Monthly."""
    if interval == "Daily":
        return df.copy()
    if interval == "Weekly":
        out = (df.set_index("date")
                .resample("W")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna()
                .reset_index())
        return out
    if interval == "Monthly":
        out = (df.set_index("date")
                .resample("M")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna()
                .reset_index())
        return out
    return df.copy()

def calc_kpis(df_symbol: pd.DataFrame) -> dict:
    """Calculate simple KPIs: latest_close, pct_change (vs previous), volume."""
    df_sorted = df_symbol.sort_values("date")
    latest = df_sorted.iloc[-1]
    prev = df_sorted.iloc[-2] if len(df_sorted) >= 2 else latest
    pct_change = (latest["close"] - prev["close"]) / prev["close"] * 100 if prev["close"] != 0 else 0.0
    return {"latest_close": float(latest["close"]), "pct_change": float(pct_change), "volume": int(latest.get("volume", 0))}
