# output_generate/EDA/generate_rolling_stats.py

import pandas as pd
from pathlib import Path

def main():
    # Project root (script is two folders deep)
    project_root = Path(__file__).resolve().parents[2]

    # Load final_df.parquet
    df_path = project_root / "data" / "processed" / "final_df.parquet"
    print("Loading:", df_path)
    df = pd.read_parquet(df_path)

    # Clean column names
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Output dir
    out_dir = project_root / "data" / "EDA" / "rolling"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = sorted(df["symbol"].unique().tolist())

    for sym in symbols:
        df_s = df[df["symbol"] == sym].sort_values("date").copy()

        # Rolling SMAs
        df_s["sma_7"] = df_s["close"].rolling(window=7, min_periods=1).mean()
        df_s["sma_30"] = df_s["close"].rolling(window=30, min_periods=1).mean()
        df_s["sma_100"] = df_s["close"].rolling(window=100, min_periods=1).mean()

        # Rolling Volatility (30 days)
        df_s["returns"] = df_s["close"].pct_change()
        df_s["volatility_30"] = df_s["returns"].rolling(window=30, min_periods=1).std()

        # Rolling Volume MA
        df_s["volume_ma_30"] = df_s["volume"].rolling(window=30, min_periods=1).mean()

        # Save output
        out_file = out_dir / f"{sym}_rolling.csv"
        df_s.to_csv(out_file, index=False)

        print(f"Saved rolling stats for {sym} → {out_file}")

    print("\nAll rolling statistics generated successfully!")

if __name__ == "__main__":
    main()
