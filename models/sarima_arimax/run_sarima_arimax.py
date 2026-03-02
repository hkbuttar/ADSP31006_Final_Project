#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

INPUT_CSV = Path("data/processed/merged_features_weekly.csv")
OUTPUT_ROOT = Path("models/sarima_arimax/outputs")

SARIMA_DIR = OUTPUT_ROOT / "sarima"
ARIMAX_DIR = OUTPUT_ROOT / "arimax"

EXOG_CANDIDATES = [
    "api_energy_and_lubricants",
    "api_fertilisers_and_soil_improvers",
    "api_plant_protection_products",
    "api_fresh_fruit",
    "api_fresh_vegetables",
    "fuel_petrol_price",
    "fuel_diesel_price",
    "sppi_road_freight",
    "shock_2021q4_2023q1",
    "post_shock",
    "week_sin",
    "week_cos",
]


@dataclass
class Metrics:
    rmse: float
    mae: float
    mase: float
    smape: float
    mape: float


def ensure_dirs() -> None:
    for model_dir in (SARIMA_DIR, ARIMAX_DIR):
        (model_dir / "forecasts").mkdir(parents=True, exist_ok=True)


def sanitize(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch in (" ", "/", "(", ")"):
            keep.append("_")
    clean = "".join(keep).strip("_")
    while "__" in clean:
        clean = clean.replace("__", "_")
    return clean or "unknown"


def mase(actual: np.ndarray, pred: np.ndarray, train: np.ndarray) -> float:
    if len(train) < 2:
        return float("nan")
    denom = np.mean(np.abs(np.diff(train)))
    if denom == 0 or np.isnan(denom):
        return float("nan")
    return float(np.mean(np.abs(actual - pred)) / denom)


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = actual != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100.0)


def smape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.abs(actual) + np.abs(pred)
    mask = denom != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(200.0 * np.abs(actual[mask] - pred[mask]) / denom[mask]))


def compute_metrics(actual: np.ndarray, pred: np.ndarray, train: np.ndarray) -> Metrics:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return Metrics(
        rmse=float(np.sqrt(np.mean((actual - pred) ** 2))),
        mae=float(np.mean(np.abs(actual - pred))),
        mase=mase(actual, pred, train),
        smape=smape(actual, pred),
        mape=mape(actual, pred),
    )


def split_train_test(df_one: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "split" in df_one.columns and set(df_one["split"].dropna().unique()) >= {"train", "test"}:
        train = df_one[df_one["split"] == "train"].copy()
        test = df_one[df_one["split"] == "test"].copy()
        if len(train) > 0 and len(test) > 0:
            return train, test
    test_weeks = min(52, max(1, len(df_one) // 5))
    train = df_one.iloc[:-test_weeks].copy()
    test = df_one.iloc[-test_weeks:].copy()
    return train, test


def fit_sarima(train: pd.Series, horizon: int):
    model = SARIMAX(
        endog=train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 52),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    pred = fit.get_forecast(steps=horizon).predicted_mean
    return fit, np.asarray(pred, dtype=float)


def fit_arimax(train_y: pd.Series, train_x: pd.DataFrame, test_x: pd.DataFrame):
    model = SARIMAX(
        endog=train_y,
        exog=train_x,
        order=(1, 1, 1),
        seasonal_order=(0, 0, 0, 0),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    pred = fit.get_forecast(steps=len(test_x), exog=test_x).predicted_mean
    return fit, np.asarray(pred, dtype=float)


def write_forecast_file(
    out_path: Path,
    commodity: str,
    dates: pd.Series,
    actual: np.ndarray,
    pred: np.ndarray,
    model_name: str,
) -> None:
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(dates).dt.date,
            "commodity": commodity,
            "actual": actual,
            "predicted": pred,
            "residual": actual - pred,
            "model": model_name,
        }
    )
    out.to_csv(out_path, index=False)


def main() -> None:
    ensure_dirs()
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
    req = {"date", "commodity", "target_price"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    exog_cols = [c for c in EXOG_CANDIDATES if c in df.columns]

    sarima_rows: List[Dict] = []
    arimax_rows: List[Dict] = []
    status_rows: List[Dict] = []

    for commodity, g in df.sort_values("date").groupby("commodity"):
        g = g.sort_values("date").copy()
        train_df, test_df = split_train_test(g)

        if len(train_df) < 110 or len(test_df) < 4:
            status_rows.append(
                {
                    "commodity": commodity,
                    "status": "skipped",
                    "reason": f"insufficient rows train={len(train_df)} test={len(test_df)}",
                }
            )
            continue

        y_train = pd.to_numeric(train_df["target_price"], errors="coerce").astype(float)
        y_test = pd.to_numeric(test_df["target_price"], errors="coerce").astype(float)

        if y_train.isna().any() or y_test.isna().any():
            status_rows.append(
                {
                    "commodity": commodity,
                    "status": "skipped",
                    "reason": "target_price has NA values after coercion",
                }
            )
            continue

        safe = sanitize(commodity)

        try:
            sarima_fit, sarima_pred = fit_sarima(y_train, len(y_test))
            sarima_m = compute_metrics(y_test.values, sarima_pred, y_train.values)
            write_forecast_file(
                SARIMA_DIR / "forecasts" / f"{safe}.csv",
                commodity,
                test_df["date"],
                y_test.values,
                sarima_pred,
                "SARIMA(1,1,1)(1,1,1,52)",
            )
            sarima_rows.append(
                {
                    "commodity": commodity,
                    "model": "SARIMA(1,1,1)(1,1,1,52)",
                    "RMSE": sarima_m.rmse,
                    "MAE": sarima_m.mae,
                    "MASE": sarima_m.mase,
                    "sMAPE": sarima_m.smape,
                    "MAPE": sarima_m.mape,
                    "AIC": float(sarima_fit.aic) if sarima_fit.aic is not None else math.nan,
                    "BIC": float(sarima_fit.bic) if sarima_fit.bic is not None else math.nan,
                    "train_rows": len(train_df),
                    "test_rows": len(test_df),
                }
            )
        except Exception as exc:  # noqa: BLE001
            status_rows.append({"commodity": commodity, "status": "sarima_failed", "reason": str(exc)})

        if exog_cols:
            try:
                x_train = train_df[exog_cols].apply(pd.to_numeric, errors="coerce")
                x_test = test_df[exog_cols].apply(pd.to_numeric, errors="coerce")
                x_train = x_train.ffill().bfill()
                x_test = x_test.ffill().bfill()

                arimax_fit, arimax_pred = fit_arimax(y_train, x_train, x_test)
                arimax_m = compute_metrics(y_test.values, arimax_pred, y_train.values)
                write_forecast_file(
                    ARIMAX_DIR / "forecasts" / f"{safe}.csv",
                    commodity,
                    test_df["date"],
                    y_test.values,
                    arimax_pred,
                    "ARIMAX(1,1,1)+xreg",
                )
                arimax_rows.append(
                    {
                        "commodity": commodity,
                        "model": "ARIMAX(1,1,1)+xreg",
                        "RMSE": arimax_m.rmse,
                        "MAE": arimax_m.mae,
                        "MASE": arimax_m.mase,
                        "sMAPE": arimax_m.smape,
                        "MAPE": arimax_m.mape,
                        "AIC": float(arimax_fit.aic) if arimax_fit.aic is not None else math.nan,
                        "BIC": float(arimax_fit.bic) if arimax_fit.bic is not None else math.nan,
                        "n_exog": len(exog_cols),
                        "exog_columns": "|".join(exog_cols),
                        "train_rows": len(train_df),
                        "test_rows": len(test_df),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                status_rows.append({"commodity": commodity, "status": "arimax_failed", "reason": str(exc)})
        else:
            status_rows.append({"commodity": commodity, "status": "arimax_skipped", "reason": "no exog columns found"})

    pd.DataFrame(sarima_rows).sort_values("commodity").to_csv(SARIMA_DIR / "metrics.csv", index=False)
    pd.DataFrame(arimax_rows).sort_values("commodity").to_csv(ARIMAX_DIR / "metrics.csv", index=False)
    pd.DataFrame(status_rows).to_csv(OUTPUT_ROOT / "run_status.csv", index=False)

    print(f"Wrote: {SARIMA_DIR / 'metrics.csv'}")
    print(f"Wrote: {ARIMAX_DIR / 'metrics.csv'}")
    print(f"Wrote: {OUTPUT_ROOT / 'run_status.csv'}")


if __name__ == "__main__":
    main()
