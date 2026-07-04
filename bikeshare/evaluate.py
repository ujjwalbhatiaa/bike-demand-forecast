"""Leakage-safe evaluation utilities.

A random train/test split on time-series data lets the model peek at the
future (rows from the same day land in both sets), which inflates scores.
We split **chronologically**: train on the past, test on the most recent
slice — the honest way to evaluate a forecasting model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def time_split(
    df: pd.DataFrame, test_fraction: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a chronologically sorted frame into (train, test).

    The final ``test_fraction`` of rows — the most recent observations —
    become the test set.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if not df["dteday"].is_monotonic_increasing:
        raise ValueError("dataframe must be sorted chronologically")

    cut = int(round(len(df) * (1.0 - test_fraction)))
    train, test = df.iloc[:cut], df.iloc[cut:]
    if train.empty or test.empty:
        raise ValueError("split produced an empty train or test set")
    return train, test


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """RMSE, MAE and R² in one dict (computed with numpy, no sklearn needed)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    residuals = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}
