"""
model.py

Trains and compares two regression models for forecasting:
1. Linear Regression - interpretable baseline
2. Random Forest Regressor - captures non-linear patterns/interactions

Uses a chronological train/test split (never shuffle time series data,
since that leaks future information into training).
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def fit_trend(train_df, day_index_col="day_index", target_col="sales"):
    """
    Fits a simple linear trend line (sales ~ day_index) on the training set.

    Why: Tree-based models (Random Forest) cannot extrapolate a raw
    numeric trend beyond the range of values seen during training - they
    just predict the value of whichever leaf a new point falls into. On a
    growing sales series, this causes systematic underprediction on future
    data. The standard fix is to detrend the target with a simple model
    first, let the more flexible model (RF) learn the leftover residual
    pattern (seasonality, promo effects, day-of-week effects), and then
    add the extrapolated trend back onto the residual prediction.
    """
    trend_model = LinearRegression()
    X_trend = train_df[[day_index_col]].values
    y = train_df[target_col].values
    trend_model.fit(X_trend, y)
    return trend_model


def chronological_split(df, test_size=0.2):
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df


def train_linear_regression(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return model, scaler


def train_random_forest(X_train, y_train, n_estimators=300, random_state=42):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=8,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model
