"""Feature engineering for the hourly demand model.

Key ideas
---------
* **Cyclical encoding** — hour 23 and hour 0 are adjacent, but as plain
  integers they are maximally far apart. Encoding each cyclic variable as
  (sin, cos) pairs puts midnight next to 11 pm, December next to January.
* **Real units** — the UCI file normalizes weather columns; we add back
  Celsius / % / km/h copies so plots and feature importances are readable.
* **No leakage** — ``casual`` and ``registered`` sum exactly to the target
  ``cnt``, so they are never used as features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Denormalization constants from the dataset's Readme.txt.
TEMP_MAX_C = 41.0
ATEMP_MAX_C = 50.0
WINDSPEED_MAX = 67.0

#: Columns fed to the models, in order.
FEATURE_COLUMNS = [
    "yr", "season", "holiday", "workingday", "weathersit",
    "temp_c", "feels_like_c", "humidity_pct", "windspeed_kmh",
    "hr_sin", "hr_cos", "mnth_sin", "mnth_cos", "weekday_sin", "weekday_cos",
]

TARGET_COLUMN = "cnt"


def _cyclical(series: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    radians = 2.0 * np.pi * series / period
    return np.sin(radians), np.cos(radians)


def add_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with real-unit weather and cyclical time columns."""
    out = df.copy()

    out["temp_c"] = out["temp"] * TEMP_MAX_C
    out["feels_like_c"] = out["atemp"] * ATEMP_MAX_C
    out["humidity_pct"] = out["hum"] * 100.0
    out["windspeed_kmh"] = out["windspeed"] * WINDSPEED_MAX

    out["hr_sin"], out["hr_cos"] = _cyclical(out["hr"], 24)
    out["mnth_sin"], out["mnth_cos"] = _cyclical(out["mnth"] - 1, 12)
    out["weekday_sin"], out["weekday_cos"] = _cyclical(out["weekday"], 7)

    return out


def build_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Turn a (validated) raw frame into the model matrix X and target y."""
    enriched = add_engineered_columns(df)
    X = enriched[FEATURE_COLUMNS].astype(float)
    y = enriched[TARGET_COLUMN].astype(float)
    if X.isna().any().any():
        raise ValueError("feature matrix contains NaN values")
    return X, y
