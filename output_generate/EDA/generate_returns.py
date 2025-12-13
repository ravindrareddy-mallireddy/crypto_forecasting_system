# output_generate/EDA/generate_returns.py

import pandas as pd
from pathlib import Path

def main():
    # Project root: (script is 2 levels deep)
    project_root = Path(__file__).resolve().parents[2]

    # Input dataset
    df_path = project_root / "data" / "processed" / "final_df.parquet"

    # Output folder for returns
    out_dir = project_root / "data" / "EDA" / "returns"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading:", df_path)
    df = pd.read_parquet(df_path)

    # ensure consistency
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    symbols = sorted(df["symbol"].unique().tolist())

    for sym in symbols:
        df_s = df[df["symbol"] == sym].sort_values("date").copy()

        # compute returns
        df_s["returns"] = df_s["close"].pct_change()

        # output file for that coin
        out_file = out_dir / f"{sym}_returns.csv"
        df_s.to_csv(out_file, index=False)

        print(f"Saved returns for {sym} → {out_file}")

    print("\nAll coin returns generated successfully!")

if __name__ == "__main__":
    main()
