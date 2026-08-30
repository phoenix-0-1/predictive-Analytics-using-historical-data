# Predictive Sales Forecasting

A complete, end-to-end predictive analytics project: clean historical sales
data, engineer time-series features, train regression models, evaluate
their accuracy, and visualize the forecasts.

## Project Goal

Forecast future daily sales based on historical trends, weekly/yearly
seasonality, and promotional activity — using regression models augmented
with time-series feature engineering (lags, rolling averages, trend
decomposition).

## Project Structure

```
predictive-sales-forecasting/
├── data/
│   └── raw/
│       └── sales_data.csv        # historical dataset (generated or your own)
├── src/
│   ├── generate_data.py          # synthetic data generator (swap for real data)
│   ├── preprocessing.py          # cleaning: missing values, dupes, outliers
│   ├── features.py               # feature engineering (lags, rolling means, calendar)
│   ├── model.py                  # model training (Linear Regression, Random Forest)
│   └── evaluate.py               # accuracy metrics (MAE, RMSE, MAPE, R²)
├── outputs/
│   ├── plots/                    # generated visualizations
│   └── metrics/
│       └── metrics.json          # saved accuracy scores
├── main.py                       # runs the full pipeline end-to-end
├── requirements.txt
└── README.md
```

## How It Works

1. **Data** — `src/generate_data.py` builds a realistic synthetic daily
   sales series (trend + weekly/yearly seasonality + random promotions +
   noise). To use your own data, just replace `data/raw/sales_data.csv`
   with a CSV containing `date` and `sales` columns — the rest of the
   pipeline works unchanged.

2. **Preprocessing** (`src/preprocessing.py`) — parses dates, fills any
   missing calendar days, forward/back-fills missing sales values, and
   clips outliers using the IQR method.

3. **Feature Engineering** (`src/features.py`) — builds the inputs a
   regression model needs to "see" time-series structure:
   - Calendar features: day of week, month, weekend flag
   - Cyclical yearly seasonality: `sin`/`cos` encoding of day-of-year
   - Lag features: sales 1, 7, and 14 days ago
   - Rolling averages: 7-day and 30-day trailing mean

4. **Modeling** (`src/model.py`) — trains two models on a **chronological**
   80/20 train/test split (never shuffle time series data):
   - **Linear Regression** — interpretable baseline, extrapolates the
     long-term trend naturally.
   - **Random Forest** — captures non-linear seasonal/promotional
     interactions. Trained on the **detrended residual** (sales minus a
     fitted linear trend line), because tree-based models cannot
     extrapolate a raw trend beyond the range seen in training — a common
     real-world pitfall. The trend is added back after prediction.

5. **Evaluation** (`src/evaluate.py`) — MAE, RMSE, MAPE, and R² on the
   held-out test period.

6. **Visualization** — three plots saved to `outputs/plots/`:
   - `actual_vs_predicted.png` — forecast vs ground truth over the test period
   - `residuals.png` — error pattern over time (checks for bias/drift)
   - `feature_importance.png` — which features the Random Forest relies on

## Setup & Usage

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd predictive-sales-forecasting

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python main.py
```

Output:
- Cleaned/engineered data is processed in-memory (raw data cached in `data/raw/`)
- Plots saved to `outputs/plots/`
- Metrics saved to `outputs/metrics/metrics.json`

## Using Your Own Dataset

Replace `data/raw/sales_data.csv` with any CSV that has at minimum:

| date       | sales  |
|------------|--------|
| 2024-01-01 | 412.50 |
| 2024-01-02 | 398.10 |

An optional `is_promotion` (0/1) column is also used if present. Delete
the existing CSV and re-run `main.py`, or point `RAW_PATH` in `main.py`
to your file.

## Key Learnings Demonstrated

- Time-series-aware preprocessing (no shuffling, gap-filling, outlier clipping)
- Turning a raw time series into a supervised-learning problem via lag/rolling
  features
- Why tree-based models struggle with trending data, and how to fix it
  (detrend-then-model)
- Standard forecast accuracy metrics and how to interpret them
- Visual diagnostics (residual plots, feature importance) as part of model
  evaluation, not just a single accuracy number

## Extending This Project

- Swap in `statsmodels` (ARIMA/SARIMA) or `Prophet` for a pure time-series
  comparison against the regression approach
- Add hyperparameter tuning (`GridSearchCV`) for the Random Forest
- Add confidence intervals / prediction intervals to the forecast
- Deploy the trained model behind a small Flask/FastAPI endpoint
