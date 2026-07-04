"""Load and validate the UCI Bike Sharing (hourly) dataset.

Dataset: Fanaee-T, Hadi, and Gama, Joao (2013). Event labeling combining
ensemble detectors and background knowledge. Progress in Artificial
Intelligence. https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Every column the raw hour.csv must contain.
EXPECTED_COLUMNS = [
    "instant", "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed",
    "casual", "registered", "cnt",
]

# Inclusive valid ranges for the coded/normalized columns.
VALID_RANGES = {
    "season": (1, 4),
    "yr": (0, 1),
    "mnth": (1, 12),
    "hr": (0, 23),
    "holiday": (0, 1),
    "weekday": (0, 6),
    "workingday": (0, 1),
    "weathersit": (1, 4),
    "temp": (0.0, 1.0),
    "atemp": (0.0, 1.0),
    "hum": (0.0, 1.0),
    "windspeed": (0.0, 1.0),
}


class DataValidationError(ValueError):
    """Raised when the raw CSV does not look like the UCI hourly dataset."""


def load_hourly(path: str | Path) -> pd.DataFrame:
    """Read hour.csv, validate schema and value ranges, return a DataFrame.

    The frame is sorted chronologically (by date, then hour), which the
    time-based splitting in :mod:`bikeshare.evaluate` relies on.
    """
    df = pd.read_csv(path, parse_dates=["dteday"])

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing expected columns: {missing}")

    if df[EXPECTED_COLUMNS].isna().any().any():
        raise DataValidationError("dataset contains missing values")

    for col, (lo, hi) in VALID_RANGES.items():
        bad = df[(df[col] < lo) | (df[col] > hi)]
        if not bad.empty:
            raise DataValidationError(
                f"column {col!r} has {len(bad)} values outside [{lo}, {hi}]"
            )

    # The two rider segments must sum to the target — a strong integrity check
    # and the reason casual/registered are *leakage* if used as features.
    if not (df["casual"] + df["registered"] == df["cnt"]).all():
        raise DataValidationError("casual + registered != cnt for some rows")

    return df.sort_values(["dteday", "hr"], kind="mergesort").reset_index(drop=True)
