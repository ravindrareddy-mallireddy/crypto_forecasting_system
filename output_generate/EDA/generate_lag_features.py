# output_generate/EDA/generate_lag_features.py

import pandas as pd
from pathlib import Path

def main():
    # Project root
    project_root = Path(__file__).resolve().parents[2]

    # Load dataset
    df_path = project_root / "data" / "processed" / "final_df.parquet"
    print("Loading:", df_path)

    df = pd.read_parquet(df_path)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Output directory
    out_dir = project_root / "data" / "EDA" / "lag"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = sorted(df["symbol"].unique().tolist())

    for sym in symbols:
        df_s = df[df["symbol"] == sym].sort_values("date").copy()

        # Lag features
        df_s["lag_1"] = df_s["close"].shift(1)
        df_s["lag_7"] = df_s["close"].shift(7)
        df_s["lag_30"] = df_s["close"].shift(30)

        # Save only relevant columns
        df_out = df_s[["date", "symbol", "close", "lag_1", "lag_7", "lag_30"]]

        out_file = out_dir / f"{sym}_lags.csv"
        df_out.to_csv(out_file, index=False)

        print(f"Saved lag features for {sym} → {out_file}")

    print("\nAll lag feature files generated successfully!")

if __name__ == "__main__":
    main()
