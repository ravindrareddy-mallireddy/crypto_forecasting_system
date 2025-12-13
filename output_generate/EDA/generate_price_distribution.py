# output_generate/EDA/generate_price_distribution.py

import pandas as pd
from pathlib import Path

def main():
    # Resolve project root safely
    project_root = Path(__file__).resolve().parents[2]

    # Source data
    df_path = project_root / "data" / "processed" / "final_df.parquet"
    print("Loading:", df_path)

    df = pd.read_parquet(df_path)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Output directory
    out_dir = project_root / "data" / "EDA" / "distributions" / "price"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = sorted(df["symbol"].unique().tolist())

    for sym in symbols:
        df_s = df[df["symbol"] == sym].sort_values("date").copy()

        # Only keep what is needed
        df_out = df_s[["date", "symbol", "close"]]

        out_file = out_dir / f"{sym}_price_distribution.csv"
        df_out.to_csv(out_file, index=False)

        print(f"Saved price distribution for {sym} → {out_file}")

    print("\nAll price distribution files generated successfully!")

if __name__ == "__main__":
    main()
