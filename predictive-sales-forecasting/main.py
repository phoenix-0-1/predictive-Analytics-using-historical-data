"""
main.py

End-to-end predictive analytics pipeline:
1. Generate/load historical data
2. Clean and preprocess it
3. Engineer time-series features
4. Train Linear Regression and Random Forest models
5. Evaluate accuracy on a held-out chronological test set
6. Visualize predictions vs actuals, residuals, and feature importance

Run with:
    python main.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import pandas as pd

from src.generate_data import generate_sales_data
from src.preprocessing import clean_data
from src.features import build_features, FEATURE_COLUMNS, TARGET_COLUMN
from src.model import chronological_split, train_linear_regression, train_random_forest, fit_trend
from src.evaluate import compute_metrics

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "data" / "raw" / "sales_data.csv"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
METRICS_PATH = BASE_DIR / "outputs" / "metrics" / "metrics.json"


def step_1_get_data():
    print("\n[1/5] Loading historical data...")
    if RAW_PATH.exists():
        df = pd.read_csv(RAW_PATH, parse_dates=["date"])
        print(f"  Loaded existing dataset: {RAW_PATH} ({len(df)} rows)")
    else:
        df = generate_sales_data()
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_PATH, index=False)
        print(f"  No dataset found, generated synthetic data -> {RAW_PATH}")
    return df


def step_2_clean(df):
    print("\n[2/5] Cleaning and preprocessing...")
    df_clean = clean_data(df)
    print(f"  Clean dataset: {len(df_clean)} rows, "
          f"{df_clean['date'].min().date()} to {df_clean['date'].max().date()}")
    return df_clean


def step_3_features(df_clean):
    print("\n[3/5] Engineering features...")
    df_feat = build_features(df_clean)
    print(f"  Feature matrix: {df_feat.shape[0]} rows x {len(FEATURE_COLUMNS)} features")
    return df_feat


def step_4_train(df_feat):
    print("\n[4/5] Training models (chronological 80/20 split)...")
    train_df, test_df = chronological_split(df_feat, test_size=0.2)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    # --- Linear Regression: trained directly on raw sales ---
    # (a linear model extrapolates trend fine on its own since it fits a
    # global slope; no detrending needed)
    lr_model, scaler = train_linear_regression(X_train, y_train)
    lr_preds = lr_model.predict(scaler.transform(X_test))

    # --- Random Forest: trained on the DETRENDED residual ---
    # 1) Fit a simple linear trend on day_index -> sales using train data
    trend_model = fit_trend(train_df)
    trend_train = trend_model.predict(train_df[["day_index"]].values)
    trend_test = trend_model.predict(test_df[["day_index"]].values)

    # 2) Train RF to predict the leftover residual (seasonality, promos, etc.)
    residual_train = y_train.values - trend_train
    rf_model = train_random_forest(X_train, residual_train)

    # 3) Add the extrapolated trend back to get the final sales prediction
    rf_residual_preds = rf_model.predict(X_test)
    rf_preds = rf_residual_preds + trend_test

    print(f"  Train size: {len(train_df)}  Test size: {len(test_df)}")
    return {
        "test_df": test_df,
        "y_test": y_test,
        "lr_preds": lr_preds,
        "rf_preds": rf_preds,
        "rf_model": rf_model,
    }


def step_5_evaluate_and_visualize(results):
    print("\n[5/5] Evaluating and visualizing...")
    y_test = results["y_test"].values
    dates = results["test_df"]["date"].values

    lr_metrics = compute_metrics(y_test, results["lr_preds"])
    rf_metrics = compute_metrics(y_test, results["rf_preds"])

    print("  Linear Regression:", lr_metrics)
    print("  Random Forest:    ", rf_metrics)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump({"linear_regression": lr_metrics, "random_forest": rf_metrics}, f, indent=2)
    print(f"  Metrics saved -> {METRICS_PATH}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Actual vs Predicted ---
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_test, label="Actual", color="black", linewidth=1.5)
    plt.plot(dates, results["lr_preds"], label="Linear Regression", linestyle="--")
    plt.plot(dates, results["rf_preds"], label="Random Forest", linestyle="--")
    plt.title("Sales Forecast: Actual vs Predicted (Test Period)")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # --- Plot 2: Residuals (Random Forest) ---
    residuals = y_test - results["rf_preds"]
    plt.figure(figsize=(12, 4))
    plt.scatter(dates, residuals, alpha=0.6, s=15)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Random Forest Residuals Over Time")
    plt.xlabel("Date")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "residuals.png", dpi=150)
    plt.close()

    # --- Plot 3: Feature Importance (Random Forest) ---
    importances = results["rf_model"].feature_importances_
    order = importances.argsort()[::-1]
    plt.figure(figsize=(9, 5))
    plt.barh(
        [FEATURE_COLUMNS[i] for i in order][::-1],
        [importances[i] for i in order][::-1],
        color="steelblue",
    )
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=150)
    plt.close()

    print(f"  Plots saved -> {PLOTS_DIR}")


def main():
    df = step_1_get_data()
    df_clean = step_2_clean(df)
    df_feat = step_3_features(df_clean)
    results = step_4_train(df_feat)
    step_5_evaluate_and_visualize(results)
    print("\nDone. See outputs/plots/ and outputs/metrics/metrics.json")


if __name__ == "__main__":
    main()
