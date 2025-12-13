# output_generate/EDA/generate_seasonality.py

import pandas as pd
from pathlib import Path

def compute_monthly_returns(df_s):
    """Compute month-end returns."""
    monthly = df_s.set_index("date")["close"].resample("M").last().pct_change().dropna()
    monthly = monthly.reset_index()
    monthly.columns = ["date", "monthly_return"]
    return monthly

def compute_day_of_week_returns(df_s):
    """Compute average day-of-week returns."""
    df_s = df_s.copy()
    df_s["returns"] = df_s["close"].pct_change()
    df_s["dow"] = df_s["date"].dt.day_name()

    dow_avg = (
        df_s.groupby("dow")["returns"]
        .mean()
        .reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        .dropna()
        .reset_index()
    )

    dow_avg.columns = ["dow", "avg_return"]
    return dow_avg

def main():
    # Project root
    project_root = Path(__file__).resolve().parents[2]

    # Load final_df.parquet
    df_path = project_root / "data" / "processed" / "final_df.parquet"
    print("Loading:", df_path)
    df = pd.read_parquet(df_path)

    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Output folders
    out_dir = project_root / "data" / "EDA" / "seasonality"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = sorted(df["symbol"].unique().tolist())

    for sym in symbols:
        df_s = df[df["symbol"] == sym].sort_values("date")

        # Compute both seasonality outputs
        monthly = compute_monthly_returns(df_s)
        dow = compute_day_of_week_returns(df_s)

        # Save files
        out_file_month = out_dir / f"{sym}_monthly_returns.csv"
        out_file_dow = out_dir / f"{sym}_dow_returns.csv"

        monthly.to_csv(out_file_month, index=False)
        dow.to_csv(out_file_dow, index=False)

        print(f"Saved seasonality files for {sym} → {out_file_month}, {out_file_dow}")

    print("\nAll seasonality datasets generated successfully!")

if __name__ == "__main__":
    main()
