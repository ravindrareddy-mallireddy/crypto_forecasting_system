# output_generate/EDA/generate_volatility_clustering.py

import pandas as pd
from pathlib import Path

def main():
    # Project root (script lives 2 folders deep)
    project_root = Path(__file__).resolve().parents[2]

    # Load final dataset
    df_path = project_root / "data" / "processed" / "final_df.parquet"
    print("Loading:", df_path)

    df = pd.read_parquet(df_path)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Output folder
    out_dir = project_root / "data" / "EDA" / "volatility"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = sorted(df["symbol"].unique().tolist())

    for sym in symbols:
        df_s = df[df["symbol"] == sym].sort_values("date").copy()

        # Daily returns
        df_s["returns"] = df_s["close"].pct_change()

        # Volatility clustering measure — returns squared
        df_s["returns_squared"] = df_s["returns"] ** 2

        # Keep minimal required columns
        df_out = df_s[["date", "symbol", "returns", "returns_squared"]]

        # Save
        out_file = out_dir / f"{sym}_returns_squared.csv"
        df_out.to_csv(out_file, index=False)

        print(f"Saved volatility clustering data for {sym} → {out_file}")

    print("\nAll volatility clustering outputs generated successfully!")

if __name__ == "__main__":
    main()
