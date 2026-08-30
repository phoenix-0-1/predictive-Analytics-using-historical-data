"""
features.py

Turns a clean date/sales dataframe into a supervised-learning feature matrix:
- Calendar features (day of week, month, day of year, is_weekend)
- Trend feature (day index since start)
- Lag features (sales 1, 7, 14 days ago)
- Rolling statistics (7-day and 30-day rolling mean)

Lag/rolling features are what let a plain regression model capture
time-series dynamics (autocorrelation) without needing ARIMA.
"""

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Trend
    df["day_index"] = np.arange(len(df))

    # Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encoding for day_of_year (captures yearly seasonality smoothly)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # Lag features
    for lag in [1, 7, 14]:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    # Rolling stats (shifted by 1 to avoid leaking the current day's value)
    df["rolling_mean_7"] = df["sales"].shift(1).rolling(window=7).mean()
    df["rolling_mean_30"] = df["sales"].shift(1).rolling(window=30).mean()

    # Drop rows with NaNs created by lag/rolling windows
    df = df.dropna().reset_index(drop=True)

    return df


# Note: "day_index" (raw linear trend) is deliberately excluded from the
# shared feature set. Tree-based models like Random Forest cannot
# extrapolate a raw numeric trend beyond the range seen in training data,
# which tanks accuracy on any future/test period. Lag and rolling-mean
# features capture the trend implicitly (they "move" with the series),
# so both linear and tree-based models can use the same feature set fairly.
FEATURE_COLUMNS = [
    "day_of_week", "month", "is_weekend",
    "doy_sin", "doy_cos", "is_promotion",
    "lag_1", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_mean_30",
]

TARGET_COLUMN = "sales"
