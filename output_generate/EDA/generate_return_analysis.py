# output_generate/EDA/generate_return_analysis.py

import pandas as pd
from pathlib import Path

def main():
    # Project root
    project_root = Path(__file__).resolve().parents[2]

    # Input dataset
    df_path = project_root / "data" / "processed" / "final_df.parquet"
    print("Loading:", df_path)

    df = pd.read_parquet(df_path)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Output folder
    out_dir = project_root / "data" / "EDA" / "returns"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = sorted(df["symbol"].unique().tolist())

    for sym in symbols:
        df_s = df[df["symbol"] == sym].sort_values("date").copy()

        # Daily returns
        df_s["returns"] = df_s["close"].pct_change().fillna(0)

        # Cumulative returns
        df_s["cumulative_returns"] = (1 + df_s["returns"]).cumprod() - 1

        # Drawdown
        running_max = df_s["close"].cummax()
        df_s["drawdown"] = (df_s["close"] - running_max) / running_max

        # Save cumulative returns
        out_file_cum = out_dir / f"{sym}_cumulative_returns.csv"
        df_s[["date", "symbol", "close", "returns", "cumulative_returns"]].to_csv(out_file_cum, index=False)

        # Save drawdown
        out_file_dd = out_dir / f"{sym}_drawdown.csv"
        df_s[["date", "symbol", "close", "drawdown"]].to_csv(out_file_dd, index=False)

        print(f"Saved cumulative & drawdown for {sym}")

    print("\nReturn analysis files generated successfully!")

if __name__ == "__main__":
    main()
