# output_generate/EDA/generate_outliers.py

import pandas as pd
from pathlib import Path

def compute_outliers_iqr(df_s, col="close"):
    """Return outliers and bounds using IQR."""
    s = df_s[col].dropna()

    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df_s[(df_s[col] < lower) | (df_s[col] > upper)].copy()
    outliers["lower_bound"] = lower
    outliers["upper_bound"] = upper

    return outliers, lower, upper

def main():
    # Project root
    project_root = Path(__file__).resolve().parents[2]

    # Input data
    df_path = project_root / "data" / "processed" / "final_df.parquet"
    print("Loading:", df_path)
    df = pd.read_parquet(df_path)

    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Output folder
    out_dir = project_root / "data" / "EDA" / "outliers"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = sorted(df["symbol"].unique().tolist())

    for sym in symbols:
        df_s = df[df["symbol"] == sym].sort_values("date").copy()

        outliers, low, high = compute_outliers_iqr(df_s, col="close")

        out_file = out_dir / f"{sym}_outliers.csv"
        outliers.to_csv(out_file, index=False)

        print(f"Saved outliers for {sym} → {out_file} (bounds: {low:.2f}, {high:.2f})")

    print("\nAll outlier files generated successfully!")

if __name__ == "__main__":
    main()
