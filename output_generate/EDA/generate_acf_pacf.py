# output_generate/EDA/generate_acf_pacf.py

import pandas as pd
from pathlib import Path
from statsmodels.tsa.stattools import acf, pacf

def main():
    # Project root
    project_root = Path(__file__).resolve().parents[2]

    # Load final_df.parquet
    df_path = project_root / "data" / "processed" / "final_df.parquet"
    print("Loading:", df_path)

    df = pd.read_parquet(df_path)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Output folder
    out_dir = project_root / "data" / "EDA" / "lag"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = sorted(df["symbol"].unique().tolist())

    # Number of lags to compute
    max_lags = 40

    for sym in symbols:
        df_s = df[df["symbol"] == sym].sort_values("date").copy()

        # Use close price for ACF/PACF
        series = df_s["close"].dropna()

        # ACF values
        acf_vals = acf(series, nlags=max_lags)
        acf_df = pd.DataFrame({"lag": range(len(acf_vals)), "value": acf_vals})
        acf_file = out_dir / f"{sym}_acf.csv"
        acf_df.to_csv(acf_file, index=False)

        # PACF values
        pacf_vals = pacf(series, nlags=max_lags, method="ywm")
        pacf_df = pd.DataFrame({"lag": range(len(pacf_vals)), "value": pacf_vals})
        pacf_file = out_dir / f"{sym}_pacf.csv"
        pacf_df.to_csv(pacf_file, index=False)

        print(f"Saved ACF & PACF for {sym}")

    print("\nAll ACF & PACF files generated successfully!")

if __name__ == "__main__":
    main()
