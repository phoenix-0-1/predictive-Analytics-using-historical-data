"""
preprocessing.py

Cleans and validates the raw historical dataset:
- Parses dates, sorts chronologically
- Handles missing values (forward-fill, since sales data is a time series)
- Removes duplicate dates
- Flags/clips obvious outliers using IQR
"""

import pandas as pd
import numpy as np


def load_raw_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Sort and de-duplicate
    df = df.sort_values("date").drop_duplicates(subset="date")

    # Ensure a continuous daily date range (fills any missing calendar days)
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_range)
    df.index.name = "date"

    # Fill missing sales values: forward-fill then back-fill as a fallback
    df["sales"] = df["sales"].ffill().bfill()
    if "is_promotion" in df.columns:
        df["is_promotion"] = df["is_promotion"].fillna(0).astype(int)

    # Outlier handling using IQR clipping (cap rather than drop, to preserve
    # the time series continuity that downstream models depend on)
    q1, q3 = df["sales"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
    df["sales"] = df["sales"].clip(lower=lower, upper=upper)

    df = df.reset_index()
    return df


def run_preprocessing(raw_path, clean_path):
    df = load_raw_data(raw_path)
    df_clean = clean_data(df)
    df_clean.to_csv(clean_path, index=False)
    print(f"Cleaned data saved -> {clean_path} ({len(df_clean)} rows)")
    return df_clean


if __name__ == "__main__":
    from pathlib import Path
    base = Path(__file__).resolve().parent.parent
    run_preprocessing(base / "data" / "raw" / "sales_data.csv",
                       base / "data" / "raw" / "sales_data_clean.csv")
