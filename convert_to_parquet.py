# convert_to_parquet.py
import pandas as pd
from pathlib import Path

csv_path = Path("data/raw/final_df.csv")
parquet_path = Path("data/processed/final_df.parquet")

if not csv_path.exists():
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

df = pd.read_csv(csv_path)

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])

df.to_parquet(parquet_path, index=False)

print("Converted final_df.csv → final_df.parquet")
print("Output file:", parquet_path.resolve())
