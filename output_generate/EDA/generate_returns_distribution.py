# output_generate/EDA/generate_returns_distribution.py

import pandas as pd
from pathlib import Path

def main():
    # Resolve project root robustly
    project_root = Path(__file__).resolve().parents[2]

    # Source: previously generated returns files
    returns_dir = project_root / "data" / "EDA" / "returns"

    # Output directory
    out_dir = project_root / "data" / "EDA" / "distributions" / "returns"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loop through all *_returns.csv files
    for file in returns_dir.glob("*_returns.csv"):
        symbol = file.stem.replace("_returns", "")
        df = pd.read_csv(file, parse_dates=["date"])

        # Keep only what is needed for distribution plots
        df_out = df[["date", "symbol", "returns"]].dropna()

        out_file = out_dir / f"{symbol}_returns_distribution.csv"
        df_out.to_csv(out_file, index=False)

        print(f"Saved returns distribution for {symbol} → {out_file}")

    print("\nAll returns distribution files generated successfully!")

if __name__ == "__main__":
    main()
