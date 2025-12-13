# output_generate/EDA/generate_missing_summary.py

import pandas as pd
from pathlib import Path

def main():
    # Project root
    project_root = Path(__file__).resolve().parents[2]

    # Input dataset
    df_path = project_root / "data" / "processed" / "final_df.parquet"
    print("Loading:", df_path)
    df = pd.read_parquet(df_path)

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Calculate missingness
    missing_count = df.isnull().sum()
    missing_pct = missing_count / len(df) * 100

    missing_df = pd.DataFrame({
        "column": missing_count.index,
        "missing_count": missing_count.values,
        "pct_missing": missing_pct.values
    })

    # Output folder
    out_dir = project_root / "data" / "EDA" / "missing"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "missing_summary.csv"
    missing_df.to_csv(out_file, index=False)

    print("Missing data summary saved to:", out_file)
    print(missing_df)

if __name__ == "__main__":
    main()
