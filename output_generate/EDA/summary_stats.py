# output_generate/EDA/generate_summary_stats.py

import pandas as pd
from pathlib import Path

def main():
    # Project root (two levels up from this script)
    project_root = Path(__file__).resolve().parents[2]

    # Input final_df file
    df_path = project_root / "data" / "processed" / "final_df.parquet"

    # Output folder for EDA summary stats
    out_dir = project_root / "data" / "EDA"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "summary_stats.csv"

    # Load dataset
    print("Loading:", df_path)
    df = pd.read_parquet(df_path)

    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    symbols = sorted(df["symbol"].unique().tolist())
    rows = []

    for sym in symbols:
        df_s = df[df["symbol"] == sym].copy()
        close = df_s["close"].dropna()

        stats = {
            "symbol": sym,
            "count": int(close.count()),
            "mean": float(close.mean()),
            "median": float(close.median()),
            "std": float(close.std()),
            "min": float(close.min()),
            "max": float(close.max()),
            "skew": float(close.skew()),
            "kurtosis": float(close.kurtosis()),
            "missing_count": int(df_s.isnull().any(axis=1).sum())
        }

        rows.append(stats)

    summary_df = pd.DataFrame(rows)

    # Save CSV
    summary_df.to_csv(out_file, index=False)
    print("Summary statistics saved to:", out_file)
    print(summary_df.head())

if __name__ == "__main__":
    main()
