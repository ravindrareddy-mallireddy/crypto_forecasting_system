# output_generate/EDA/generate_volume_analysis.py

import pandas as pd
from pathlib import Path

def main():
    # Project root
    project_root = Path(__file__).resolve().parents[2]

    # Load data
    df_path = project_root / "data" / "processed" / "final_df.parquet"
    print("Loading:", df_path)

    df = pd.read_parquet(df_path)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Output folder
    out_dir = project_root / "data" / "EDA" / "volume"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = sorted(df["symbol"].unique().tolist())

    for sym in symbols:
        df_s = df[df["symbol"] == sym].sort_values("date").copy()

        # Volume moving average (30-period)
        df_s["volume_ma_30"] = df_s["volume"].rolling(window=30, min_periods=1).mean()

        # Select only needed columns
        df_out = df_s[["date", "symbol", "volume", "volume_ma_30", "close"]]

        # Save file
        out_file = out_dir / f"{sym}_volume_stats.csv"
        df_out.to_csv(out_file, index=False)

        print(f"Saved volume analysis for {sym} → {out_file}")

    print("\nAll volume analysis datasets generated successfully!")

if __name__ == "__main__":
    main()
