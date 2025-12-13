# output_generate/EDA/generate_correlation.py

import pandas as pd
from pathlib import Path

def main():
    # Project root (script two folders deep)
    project_root = Path(__file__).resolve().parents[2]

    # Input final_df dataset
    df_path = project_root / "data" / "processed" / "final_df.parquet"
    print("Loading:", df_path)
    df = pd.read_parquet(df_path)

    # Standardize columns
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Output folder
    out_dir = project_root / "data" / "EDA" / "correlation"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = sorted(df["symbol"].unique().tolist())

    corr_cols = ["open", "high", "low", "close", "volume"]

    for sym in symbols:
        df_s = df[df["symbol"] == sym].copy()

        # compute correlation
        corr = df_s[corr_cols].corr()

        # save to CSV
        out_file = out_dir / f"{sym}_corr.csv"
        corr.to_csv(out_file)

        print(f"Saved correlation matrix for {sym} → {out_file}")

    print("\nAll OHLCV correlation matrices generated successfully!")

if __name__ == "__main__":
    main()
