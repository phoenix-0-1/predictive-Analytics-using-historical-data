"""
generate_data.py

Generates a realistic synthetic daily sales dataset with:
- A long-term upward trend
- Weekly seasonality (weekends higher)
- Yearly seasonality (holiday season boost)
- Random promotion days that boost sales
- Gaussian noise

In a real project you would replace this with your own historical
CSV (e.g. from a POS system, Google Analytics export, or ERP report).
The rest of the pipeline (preprocessing, feature engineering, modeling)
works on any dataset with a `date` and `sales` column.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "sales_data.csv"


def generate_sales_data(start_date="2021-01-01", periods=1000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start=start_date, periods=periods, freq="D")
    t = np.arange(periods)

    # 1. Long-term trend (gradual growth with slight acceleration)
    trend = 200 + 0.15 * t + 0.00005 * t**2

    # 2. Weekly seasonality: Fri/Sat/Sun higher
    day_of_week = dates.dayofweek  # Mon=0 ... Sun=6
    weekly_effect = np.where(day_of_week >= 4, 40, 0) + np.where(day_of_week == 5, 20, 0)

    # 3. Yearly seasonality: boost in Nov-Dec (holiday shopping), dip in Feb
    day_of_year = dates.dayofyear
    yearly_effect = 60 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    holiday_boost = np.where(dates.month.isin([11, 12]), 50, 0)

    # 4. Random promotion days (~8% of days), each boosts sales 15-40%
    is_promo = rng.random(periods) < 0.08
    promo_multiplier = np.where(is_promo, 1 + rng.uniform(0.15, 0.40, periods), 1.0)

    # 5. Noise
    noise = rng.normal(0, 15, periods)

    base_sales = trend + weekly_effect + yearly_effect + holiday_boost + noise
    sales = base_sales * promo_multiplier
    sales = np.clip(sales, a_min=20, a_max=None)  # sales can't be negative

    df = pd.DataFrame({
        "date": dates,
        "sales": np.round(sales, 2),
        "is_promotion": is_promo.astype(int),
    })

    return df


def main():
    df = generate_sales_data()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df)} rows of synthetic sales data -> {OUTPUT_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()
