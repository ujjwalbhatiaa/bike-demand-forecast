#!/usr/bin/env python3
"""Predict hourly bike-share demand for a single set of conditions.

The training pipeline (``train.py``) answers "how good is this model?" by
scoring it on months it has never seen. This script answers the operational
question a rebalancing team actually has: "given tomorrow's forecast, how
many bikes will we need at 8am?"

It trains the same Gradient Boosting model used in the report (fast — well
under a second, see MODEL-REPORT.md) on the *full* dataset, builds a single
feature row from the conditions you pass on the command line using the exact
same feature engineering as training (bikeshare.features), and prints the
predicted rental count.

Usage
-----
    python predict.py --season 2 --month 6 --hour 8 --weekday 1 \\
        --workingday --weather 1 --temp-c 22 --feels-like-c 24 \\
        --humidity-pct 55 --windspeed-kmh 12

    python predict.py --help          # full list of flags and their ranges
"""

from __future__ import annotations

import argparse

import pandas as pd

from bikeshare.data import VALID_RANGES, load_hourly
from bikeshare.features import (
    ATEMP_MAX_C,
    TEMP_MAX_C,
    WINDSPEED_MAX,
    add_engineered_columns,
    build_matrices,
)
from bikeshare.models import get_models

# Human-readable labels for the coded weather/season columns (from data/Readme.txt).
SEASON_LABELS = {1: "winter", 2: "spring", 3: "summer", 4: "fall"}
WEATHER_LABELS = {
    1: "clear / few clouds",
    2: "mist + cloudy",
    3: "light rain/snow",
    4: "heavy rain/snow + fog",
}


def _bounded(name: str, value: float, lo: float, hi: float) -> float:
    if not (lo <= value <= hi):
        raise argparse.ArgumentTypeError(f"--{name} must be between {lo} and {hi} (got {value})")
    return value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/hour.csv", help="training CSV (default: data/hour.csv)")
    p.add_argument("--year", type=int, choices=[2011, 2012], default=2012, help="calendar year")
    p.add_argument("--season", type=int, choices=[1, 2, 3, 4], required=True,
                    help="1=winter 2=spring 3=summer 4=fall")
    p.add_argument("--month", type=int, choices=range(1, 13), required=True, metavar="[1-12]")
    p.add_argument("--hour", type=int, choices=range(0, 24), required=True, metavar="[0-23]")
    p.add_argument("--weekday", type=int, choices=range(0, 7), required=True,
                    metavar="[0-6]", help="0=Sunday ... 6=Saturday")
    p.add_argument("--holiday", action="store_true", help="is this a public holiday")
    p.add_argument("--workingday", action="store_true", help="is this a working day (not weekend/holiday)")
    p.add_argument("--weather", type=int, choices=[1, 2, 3, 4], required=True,
                    help="1=clear 2=mist 3=light rain/snow 4=heavy rain/snow")
    p.add_argument("--temp-c", type=float, required=True, help="air temperature in Celsius")
    p.add_argument("--feels-like-c", type=float, required=True, help="apparent temperature in Celsius")
    p.add_argument("--humidity-pct", type=float, required=True, help="relative humidity, 0-100")
    p.add_argument("--windspeed-kmh", type=float, required=True, help="wind speed in km/h")
    return p.parse_args()


def build_query_row(args: argparse.Namespace) -> pd.DataFrame:
    """Turn CLI args into a one-row raw frame in the dataset's own units/schema."""
    row = {
        "instant": 0,
        "dteday": pd.Timestamp(args.year, 1, 1),  # placeholder, only yr/mnth/hr feed the model
        "season": args.season,
        "yr": 1 if args.year == 2012 else 0,
        "mnth": args.month,
        "hr": args.hour,
        "holiday": int(args.holiday),
        "weekday": args.weekday,
        "workingday": int(args.workingday),
        "weathersit": args.weather,
        "temp": args.temp_c / TEMP_MAX_C,
        "atemp": args.feels_like_c / ATEMP_MAX_C,
        "hum": args.humidity_pct / 100.0,
        "windspeed": args.windspeed_kmh / WINDSPEED_MAX,
        "casual": 0,
        "registered": 0,
        "cnt": 0,
    }
    for col, (lo, hi) in VALID_RANGES.items():
        if not (lo <= row[col] <= hi):
            raise SystemExit(f"error: derived column {col!r}={row[col]:.3f} outside valid range [{lo}, {hi}]")
    return pd.DataFrame([row])


def main() -> None:
    args = parse_args()

    df = load_hourly(args.data)
    X_train, y_train = build_matrices(df)
    model = get_models()["Gradient Boosting"]
    model.fit(X_train, y_train)

    query_raw = build_query_row(args)
    query_enriched = add_engineered_columns(query_raw)
    X_query = query_enriched[X_train.columns]
    prediction = max(0.0, float(model.predict(X_query)[0]))

    season = SEASON_LABELS[args.season]
    weather = WEATHER_LABELS[args.weather]
    print(f"Conditions: {season}, {args.temp_c:.0f}°C ({weather}), {args.hour:02d}:00, "
          f"weekday={args.weekday}, workingday={bool(args.workingday)}")
    print(f"Predicted demand: {prediction:.0f} rentals in this hour")


if __name__ == "__main__":
    main()
