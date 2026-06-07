"""Evaluation utilities for the Brent crude oil forecasting project.

This module implements transparent, manually-computed evaluation logic
for univariate price forecasts. It provides:
- `calculate_metrics` to compute RMSE, MAE and MAPE without sklearn,
- `save_metrics` to persist metrics as JSON, and
- `save_predictions` to write Date/Actual/Predicted CSVs.

All functions expect numpy/pandas inputs and are designed for
academic reproducibility and easy inspection of the calculation steps.
"""

import os
import json
import math

import numpy as np
import pandas as pd


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE, MAE and MAPE between true and predicted arrays.

    All computations are implemented manually (no sklearn) so the logic is
    transparent for academic review.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    dict
        Dictionary with keys "RMSE", "MAE", "MAPE" and float values.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # RMSE
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = math.sqrt(mse)

    # MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # MAPE: avoid division by zero by replacing zeros with np.nan
    y_true_safe = y_true.copy()
    y_true_safe[y_true_safe == 0] = np.nan
    with np.errstate(invalid='ignore', divide='ignore'):
        mape = 100.0 * np.nanmean(np.abs((y_true - y_pred) / y_true_safe))

    # Print formatted table
    print("─── Metrics ──────────────────────")
    print(f"  RMSE : {rmse:.4f} $/barrel")
    print(f"  MAE  : {mae:.4f} $/barrel")
    print(f"  MAPE : {mape:.4f} %")

    return {"RMSE": float(rmse), "MAE": float(mae), "MAPE": float(mape)}


def save_metrics(metrics: dict, filepath: str) -> None:
    """Save metrics dictionary to a JSON file.

    Parameters
    ----------
    metrics : dict
        Dictionary of metric names to numeric values.
    filepath : str
        Destination file path for the JSON output.
    """
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=4)

    print(f"Metrics saved to {filepath}")


def save_predictions(dates, y_true: np.ndarray, y_pred: np.ndarray, filepath: str) -> None:
    """Save predictions alongside actual values to a CSV file.

    Parameters
    ----------
    dates : sequence
        Iterable of date-like values corresponding to the predictions.
    y_true : np.ndarray
        Ground-truth values.
    y_pred : np.ndarray
        Predicted values.
    filepath : str
        Destination CSV file path.
    """
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Round to 4 decimals
    actual = np.round(y_true, 4)
    predicted = np.round(y_pred, 4)

    df = pd.DataFrame({"Date": dates, "Actual": actual, "Predicted": predicted})
    df.to_csv(filepath, index=False)

    print(f"Predictions saved to {filepath}")
