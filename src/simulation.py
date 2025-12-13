# src/simulation.py
"""
What-if simulation and simple recommendation logic.

Functions:
 - simulate_profit(current_price, quantity, sell_price) -> dict
 - simple_recommendation(current_price, expected_price, target_price=None) -> str
 - basic_backtest(df, strategy_fn) -> pd.DataFrame  (small utility)
"""

from typing import Callable
import pandas as pd

def simulate_profit(current_price: float, quantity: float, sell_price: float) -> dict:
    """Return cost, revenue, absolute profit and percent profit."""
    cost = current_price * quantity
    revenue = sell_price * quantity
    profit = revenue - cost
    profit_pct = (profit / cost) * 100 if cost != 0 else 0.0
    return {"cost": cost, "revenue": revenue, "profit": profit, "profit_pct": profit_pct}

def simple_recommendation(current_price: float, expected_price: float, target_price: float = None) -> str:
    """
    Basic heuristic:
      - If expected >= target_price -> BUY
      - elif expected > current_price -> HOLD
      - else -> SELL
    """
    if expected_price is None:
        return "No forecast"
    if target_price is not None and expected_price >= target_price:
        return "BUY (expected >= target)"
    if expected_price > current_price:
        return "HOLD (expected > current)"
    return "SELL (expected <= current)"

def basic_backtest(df: pd.DataFrame, signal_fn: Callable[[pd.DataFrame], pd.Series]) -> pd.DataFrame:
    """
    Simple backtest engine.
    - df must contain 'date' and 'close' columns (sorted by date).
    - signal_fn(df) should return a Series of signals: 1 (long/buy), 0 (flat), -1 (short/sell).
    Returns a DataFrame with columns: date, close, signal, position, pnl, cumulative_pnl
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    signals = signal_fn(df)
    signals = signals.reindex(df.index).fillna(0).astype(int)
    position = signals.shift(1).fillna(0)  # enter next period
    df["signal"] = signals
    df["position"] = position
    df["return"] = df["close"].pct_change().fillna(0)
    df["pnl"] = df["position"] * df["return"]
    df["cumulative_pnl"] = df["pnl"].cumsum()
    return df[["date", "close", "signal", "position", "pnl", "cumulative_pnl"]]
