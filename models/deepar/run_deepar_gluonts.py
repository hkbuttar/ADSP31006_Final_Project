#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

INPUT_CSV = Path("data/processed/merged_features_weekly.csv")
OUTPUT_DIR = Path("models/deepar/outputs")
FORECAST_DIR = OUTPUT_DIR / "forecasts"


def ensure_dirs() -> None:
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)


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


def mase(actual: np.ndarray, pred: np.ndarray, train: np.ndarray) -> float:
    if len(train) < 2:
        return float("nan")
    denom = np.mean(np.abs(np.diff(train)))
    if denom == 0 or np.isnan(denom):
        return float("nan")
    return float(np.mean(np.abs(actual - pred)) / denom)


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
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


def write_missing_dep_status(message: str) -> None:
    pd.DataFrame(
        [{"status": "not_run", "reason": message}]
    ).to_csv(OUTPUT_DIR / "run_status.csv", index=False)


def main() -> None:
    ensure_dirs()

    try:
        from gluonts.dataset.common import ListDataset
        from gluonts.torch.model.deepar import DeepAREstimator
    except Exception as exc:  # noqa: BLE001
        write_missing_dep_status(f"GluonTS/Torch import failed: {exc}")
        print(f"DeepAR not run. See {OUTPUT_DIR / 'run_status.csv'}")
        return

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, parse_dates=["date"]).sort_values(["commodity", "date"])
    req = {"date", "commodity", "target_price"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    grouped = []
    for commodity, g in df.groupby("commodity"):
        g = g.sort_values("date").copy()
        train_df, test_df = split_train_test(g)
        if len(train_df) < 80 or len(test_df) < 4:
            continue
        grouped.append((commodity, train_df, test_df))

    if not grouped:
        write_missing_dep_status("No commodities with enough train/test rows for DeepAR")
        return

    # Keep a common prediction length across commodities for one global DeepAR model.
    prediction_length = int(min(len(test_df) for _, _, test_df in grouped))
    freq = "W-FRI"

    train_entries = []
    for commodity, train_df, _ in grouped:
        y_train = pd.to_numeric(train_df["target_price"], errors="coerce").astype(float)
        if y_train.isna().any():
            continue
        train_entries.append(
            {
                "item_id": commodity,
                "start": pd.Timestamp(train_df["date"].iloc[0]),
                "target": y_train.values,
            }
        )

    if not train_entries:
        write_missing_dep_status("All train series had NA coercion issues")
        return

    train_ds = ListDataset(train_entries, freq=freq)

    estimator = DeepAREstimator(
        prediction_length=prediction_length,
        freq=freq,
        context_length=max(2 * prediction_length, 52),
        trainer_kwargs={"max_epochs": 15},
    )
    predictor = estimator.train(train_ds)

    pred_ds = ListDataset(train_entries, freq=freq)
    forecasts = list(predictor.predict(pred_ds))

    fc_by_item = {entry["item_id"]: fc for entry, fc in zip(train_entries, forecasts)}

    metric_rows: List[Dict] = []
    status_rows: List[Dict] = []

    for commodity, train_df, test_df in grouped:
        safe = sanitize(commodity)
        if commodity not in fc_by_item:
            status_rows.append({"commodity": commodity, "status": "failed", "reason": "forecast missing from predictor"})
            continue

        y_train = pd.to_numeric(train_df["target_price"], errors="coerce").astype(float).values
        y_test = pd.to_numeric(test_df["target_price"], errors="coerce").astype(float).values[:prediction_length]
        f_mean = np.asarray(fc_by_item[commodity].mean[:prediction_length], dtype=float)

        if len(y_test) != len(f_mean):
            n = min(len(y_test), len(f_mean))
            y_test = y_test[:n]
            f_mean = f_mean[:n]

        out = pd.DataFrame(
            {
                "date": pd.to_datetime(test_df["date"]).dt.date.iloc[: len(y_test)],
                "commodity": commodity,
                "actual": y_test,
                "predicted": f_mean,
                "residual": y_test - f_mean,
                "model": "DeepAR(GluonTS)",
            }
        )
        out.to_csv(FORECAST_DIR / f"{safe}.csv", index=False)

        metric_rows.append(
            {
                "commodity": commodity,
                "model": "DeepAR(GluonTS)",
                "RMSE": float(np.sqrt(np.mean((y_test - f_mean) ** 2))),
                "MAE": float(np.mean(np.abs(y_test - f_mean))),
                "MASE": mase(y_test, f_mean, y_train),
                "sMAPE": smape(y_test, f_mean),
                "MAPE": mape(y_test, f_mean),
                "prediction_length": prediction_length,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
            }
        )
        status_rows.append({"commodity": commodity, "status": "ok", "reason": ""})

    pd.DataFrame(metric_rows).sort_values("commodity").to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    pd.DataFrame(status_rows).to_csv(OUTPUT_DIR / "run_status.csv", index=False)
    print(f"Wrote: {OUTPUT_DIR / 'metrics.csv'}")
    print(f"Wrote: {OUTPUT_DIR / 'run_status.csv'}")


if __name__ == "__main__":
    main()
