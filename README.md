# ADSP31006 Final Project — Fruit & Vegetable Price Forecasting

Weekly price forecasting for 20 UK fruit and vegetable commodities using multiple time series models.

## Structure

```
data/
  raw/          # source CSVs, Excel, and ODS files
  processed/    # merged_features_weekly.csv (184 features, W-FRI frequency)
eda/            # exploratory data analysis notebooks
feature_engineering/  # builds processed dataset from raw sources
models/
  arima/        # R: auto ARIMA, saved .rds models + metrics
  holt_winters/ # R: Holt-Winters exponential smoothing
  var/          # R: Vector Autoregression
  varma/        # R: Vector ARMA
  lm_arima/     # R: Linear regression with ARIMA errors
  mamba/        # Python: MAMBA sequence model experiments
  granger_causality/  # R: Granger causality and stationarity tests
  sarima_arimax/      # Python: SARIMA and ARIMAX runners
  deepar/             # Python: DeepAR via GluonTS/Torch
  outputs/      # cross-model comparison table
requirements.txt
```

## Data

- **Target:** weekly commodity prices (`data/raw/cleaned_fruit_veg.csv`), 2015–2026
- **Exogenous:** API agricultural indices, UK road fuel prices, ONS SPPI freight index
- **Processed:** `data/processed/merged_features_weekly.csv` — lags, rolling stats, YoY deltas, cyclical calendar features, and shock indicators for the 2021Q4–2023Q1 energy crisis

## Running the Python Models

Run from the project root:

```bash
pip install -r requirements.txt
python models/sarima_arimax/run_sarima_arimax.py
python models/deepar/run_deepar_gluonts.py
```

Outputs (metrics CSVs and per-commodity forecast CSVs) are written to each model's `outputs/` directory.

## Train / Test Split

- **Train:** up to 2024-12-27
- **Test:** 2025-01-03 onward (first Friday of 2025)
