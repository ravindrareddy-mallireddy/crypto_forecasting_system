# src/charts.py
"""
Plotly chart builders for the dashboard.
Existing functions preserved and new charts added:
 - candlestick_figure(df, title)
 - line_price_figure(df, title)
 - volume_bar_figure(df, title)

New functions:
 - sma_overlay_figure(df, windows=[20,50,200], title)
 - rolling_volatility_figure(df, window=30, title)
 - drawdown_figure(df, title)
 - returns_histogram_figure(df, freq='D', title)
 - recent_activity_table_figure(df, n=20, title)
 - get_figure_by_name(df, name, **kwargs) -> convenience router
"""

from typing import List, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# -------------------------
# Basic charts (existing)
# -------------------------
def candlestick_figure(df: pd.DataFrame, title: str = "OHLC") -> go.Figure:
    df = df.copy()
    fig = go.Figure(data=[go.Candlestick(
        x=df["date"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="OHLC"
    )])
    x0 = df["date"].min() if not df.empty else None
    x1 = df["date"].max() if not df.empty else None
    fig.update_layout(title=title, xaxis_rangeslider_visible=True,
                      xaxis=dict(range=[x0, x1] if x0 is not None else None),
                      margin=dict(t=30, b=10))
    return fig

def line_price_figure(df: pd.DataFrame, title: str = "Close Price") -> go.Figure:
    df = df.copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode="lines", name="Close"))
    x0 = df["date"].min() if not df.empty else None
    x1 = df["date"].max() if not df.empty else None
    fig.update_layout(title=title, xaxis=dict(range=[x0, x1] if x0 is not None else None, rangeslider=dict(visible=True)), margin=dict(t=30, b=10))
    return fig

def volume_bar_figure(df: pd.DataFrame, title: str = "Volume") -> go.Figure:
    df = df.copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="Volume"))
    x0 = df["date"].min() if not df.empty else None
    x1 = df["date"].max() if not df.empty else None
    fig.update_layout(title=title, xaxis=dict(range=[x0, x1] if x0 is not None else None, rangeslider=dict(visible=True)), margin=dict(t=30, b=10))
    return fig

# -------------------------
# New charts
# -------------------------
def sma_overlay_figure(df: pd.DataFrame, windows: Optional[List[int]] = None, title: str = "Price + SMA") -> go.Figure:
    """
    Plot close price with simple moving averages overlay.
    windows: list of integer window lengths in periods (e.g., [20,50,200]).
    """
    if windows is None:
        windows = [20, 50, 200]
    df = df.copy()
    df = df.sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode="lines", name="Close", line=dict(width=1.8)))
    for w in windows:
        col = f"sma_{w}"
        df[col] = df["close"].rolling(window=w, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=df["date"], y=df[col], mode="lines", name=f"SMA{w}", line=dict(dash="dash")))
    x0 = df["date"].min() if not df.empty else None
    x1 = df["date"].max() if not df.empty else None
    fig.update_layout(title=title, xaxis=dict(range=[x0, x1], rangeslider=dict(visible=True)), margin=dict(t=30, b=10))
    return fig

def rolling_volatility_figure(df: pd.DataFrame, window: int = 30, title: str = "Rolling Volatility") -> go.Figure:
    """
    Rolling volatility of daily returns (std of pct change) * sqrt(periods) to annualize if desired.
    """
    df = df.copy().sort_values("date")
    returns = df["close"].pct_change().fillna(0)
    roll_std = returns.rolling(window=window, min_periods=1).std()
    # annualize by sqrt(252) if data is daily — keep as-is for weekly/monthly
    # we will not force annualization here; user can interpret
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=roll_std, mode="lines", name=f"Rolling STD ({window})"))
    fig.update_layout(title=title + f" ({window} periods)", xaxis=dict(rangeslider=dict(visible=True)), margin=dict(t=30, b=10))
    return fig

def drawdown_figure(df: pd.DataFrame, title: str = "Drawdown (underwater)") -> go.Figure:
    """
    Compute cumulative returns and drawdown (peak-to-trough). Show underwater plot (negative drawdowns).
    """
    df = df.copy().sort_values("date")
    # avoid division by zero
    prices = df["close"].astype(float)
    cum_returns = prices / prices.iloc[0] - 1.0 if len(prices) > 0 else pd.Series(dtype=float)
    running_max = prices.cummax()
    drawdown = (prices - running_max) / running_max
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=drawdown, fill="tozeroy", name="Drawdown", line=dict(color="royalblue")))
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    fig.update_layout(title=title, xaxis=dict(rangeslider=dict(visible=True)), margin=dict(t=30, b=10))
    return fig

def returns_histogram_figure(df: pd.DataFrame, freq: str = "D", title: str = "Returns Histogram") -> go.Figure:
    """
    Histogram of returns. freq = 'D' (daily), 'W' (weekly), 'M' (monthly)
    """
    df = df.copy().sort_values("date")
    if freq.upper() == "D":
        rets = df["close"].pct_change().dropna()
    else:
        # resample close price at requested frequency and compute returns
        temp = df.set_index("date")["close"].resample(freq).last().dropna()
        rets = temp.pct_change().dropna()
    if rets.empty:
        # empty figure
        fig = go.Figure()
        fig.update_layout(title=title + " (no data)", margin=dict(t=30))
        return fig
    fig = px.histogram(rets, nbins=50, marginal="box", labels={"value": "Returns"}, title=title)
    fig.update_layout(margin=dict(t=30, b=10))
    return fig

def recent_activity_table_figure(df: pd.DataFrame, n: int = 20, title: str = "Recent Activity") -> go.Figure:
    """
    Return a Plotly table showing the most recent n rows (OHLCV).
    """
    df = df.copy().sort_values("date", ascending=False).head(n)
    # show date first column (formatted)
    table_df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    table_df["date"] = table_df["date"].dt.strftime("%Y-%m-%d")
    header = list(table_df.columns)
    cells = [table_df[col].tolist() for col in header]
    fig = go.Figure(data=[go.Table(
        header=dict(values=header, fill_color="lightgrey", align="left"),
        cells=dict(values=cells, align="left", format=[None, ".4f", ".4f", ".4f", ".4f", ",d"])
    )])
    fig.update_layout(title=title, margin=dict(t=30, b=10))
    return fig

# -------------------------
# Convenience router
# -------------------------
def get_figure_by_name(df: pd.DataFrame, name: str, **kwargs) -> go.Figure:
    """
    Map a friendly name to the appropriate figure builder.
    name values:
      - "Price time series"
      - "Candlestick (OHLC)"
      - "Volume"
      - "SMA overlay"
      - "Rolling volatility"
      - "Drawdown"
      - "Returns histogram"
      - "Recent activity"
    """
    key = name.lower()
    if "candlestick" in key or "ohlc" in key:
        return candlestick_figure(df, title=name)
    if "price" in key and "sma" not in key:
        return line_price_figure(df, title=name)
    if "volume" in key and "hist" not in key:
        return volume_bar_figure(df, title=name)
    if "sma" in key or "moving" in key or "overlay" in key:
        windows = kwargs.get("windows", [20, 50, 200])
        return sma_overlay_figure(df, windows=windows, title=name)
    if "volatility" in key:
        w = kwargs.get("window", 30)
        return rolling_volatility_figure(df, window=w, title=name)
    if "drawdown" in key or "underwater" in key:
        return drawdown_figure(df, title=name)
    if "hist" in key or "returns" in key:
        freq = kwargs.get("freq", "D")
        return returns_histogram_figure(df, freq=freq, title=name)
    if "recent" in key or "activity" in key or "table" in key:
        n = kwargs.get("n", 20)
        return recent_activity_table_figure(df, n=n, title=name)
    # fallback
    return line_price_figure(df, title=name)
