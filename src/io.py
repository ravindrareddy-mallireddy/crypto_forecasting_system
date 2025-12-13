# src/io.py
"""
Dataset loading and simple I/O helpers.

Functions
---------
- load_dataset(paths=None) -> pd.DataFrame
- list_symbols(df) -> list[str]
- save_parquet(df, path) -> None
"""

from pathlib import Path
import pandas as pd

DEFAULT_CANDIDATES = [
    Path(__file__).parents[1] / "data" / "processed" / "final_df.parquet",
    Path(__file__).parents[1] / "data" / "processed" / "final_df.csv",
    Path(__file__).parents[1] / "data" / "final_df.parquet",
    Path(__file__).parents[1] / "data" / "final_df.csv",
]

REQUIRED_COLS = {"date", "symbol", "open", "high", "low", "close", "volume"}

def load_dataset(paths: list = None) -> pd.DataFrame:
    """
    Load processed dataset from one of the candidate paths.
    Raises FileNotFoundError or ValueError (if columns missing).
    """
    paths = paths or DEFAULT_CANDIDATES
    csv_kwargs = {"parse_dates": ["date"], "infer_datetime_format": True}
    found = None
    for p in paths:
        p = Path(p)
        if p.exists():
            found = p
            break
    if found is None:
        raise FileNotFoundError(f"No dataset found. Checked: {paths}")

    if found.suffix == ".parquet":
        df = pd.read_parquet(found)
    else:
        df = pd.read_csv(found, **csv_kwargs)
    # normalize columns
    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df

def list_symbols(df: pd.DataFrame) -> list:
    """Return sorted unique symbols from dataset."""
    return sorted(df["symbol"].dropna().unique().tolist())

def save_parquet(df: pd.DataFrame, path: str = None) -> Path:
    """Save dataframe to parquet under data/processed if path not given."""
    base = Path(__file__).parents[1]
    out = Path(path) if path else base / "data" / "processed" / "final_df.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out
