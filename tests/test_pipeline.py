"""Tests for the bike-demand pipeline. Run with:  python -m pytest tests/ -q"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bikeshare.data import DataValidationError, load_hourly
from bikeshare.evaluate import regression_metrics, time_split
from bikeshare.features import FEATURE_COLUMNS, add_engineered_columns, build_matrices
from bikeshare.models import get_models
from predict import build_query_row

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "hour.csv"


@pytest.fixture(scope="module")
def df():
    return load_hourly(DATA_PATH)


# ---------------------------------------------------------------- data loading
def test_loads_full_dataset(df):
    assert len(df) == 17379
    assert df["dteday"].min() == pd.Timestamp("2011-01-01")
    assert df["dteday"].max() == pd.Timestamp("2012-12-31")


def test_sorted_chronologically(df):
    key = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    assert key.is_monotonic_increasing


def test_validation_catches_missing_column(tmp_path):
    bad = pd.read_csv(DATA_PATH, nrows=50).drop(columns=["cnt"])
    p = tmp_path / "bad.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(DataValidationError):
        load_hourly(p)


def test_validation_catches_out_of_range(tmp_path):
    bad = pd.read_csv(DATA_PATH, nrows=50)
    bad.loc[0, "hr"] = 99
    p = tmp_path / "bad.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(DataValidationError):
        load_hourly(p)


# ---------------------------------------------------------------- features
def test_cyclical_encoding_wraps_midnight(df):
    enriched = add_engineered_columns(df)
    h23 = enriched.loc[enriched.hr == 23, ["hr_sin", "hr_cos"]].iloc[0]
    h0 = enriched.loc[enriched.hr == 0, ["hr_sin", "hr_cos"]].iloc[0]
    h12 = enriched.loc[enriched.hr == 12, ["hr_sin", "hr_cos"]].iloc[0]
    dist_wrap = np.hypot(h23.hr_sin - h0.hr_sin, h23.hr_cos - h0.hr_cos)
    dist_far = np.hypot(h12.hr_sin - h0.hr_sin, h12.hr_cos - h0.hr_cos)
    assert dist_wrap < dist_far  # 23:00 is closer to 00:00 than noon is


def test_real_unit_columns(df):
    enriched = add_engineered_columns(df)
    assert enriched["temp_c"].between(-10, 45).all()
    assert enriched["humidity_pct"].between(0, 100).all()


def test_matrices_shape_and_no_leakage(df):
    X, y = build_matrices(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y) == len(df)
    assert "casual" not in X.columns and "registered" not in X.columns
    assert not X.isna().any().any()


# ---------------------------------------------------------------- evaluation
def test_time_split_is_chronological(df):
    train, test = time_split(df, 0.15)
    assert train["dteday"].max() <= test["dteday"].min()
    assert len(train) + len(test) == len(df)
    assert abs(len(test) / len(df) - 0.15) < 0.01


def test_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(y, y)
    assert m["rmse"] == 0.0 and m["mae"] == 0.0 and m["r2"] == 1.0


def test_metrics_known_values():
    m = regression_metrics([0.0, 0.0], [3.0, -3.0])
    assert m["rmse"] == pytest.approx(3.0)
    assert m["mae"] == pytest.approx(3.0)


# ---------------------------------------------------------------- models
def test_models_beat_baseline_on_subset(df):
    """Smoke test: on a 3-month slice, every real model beats the mean baseline."""
    subset = df[df["dteday"] < "2011-04-01"]
    train, test = time_split(subset, 0.2)
    X_tr, y_tr = build_matrices(train)
    X_te, y_te = build_matrices(test)

    scores = {}
    for name, model in get_models().items():
        model.fit(X_tr, y_tr)
        scores[name] = regression_metrics(y_te, model.predict(X_te))["rmse"]

    for name, rmse in scores.items():
        if name != "Mean baseline":
            assert rmse < scores["Mean baseline"], f"{name} did not beat baseline"


# ---------------------------------------------------------------- predict CLI
def _args(**overrides):
    """Build an argparse.Namespace with sensible defaults for build_query_row."""
    defaults = dict(
        year=2012, season=2, month=6, hour=8, weekday=1,
        holiday=False, workingday=True, weather=1,
        temp_c=22.0, feels_like_c=24.0, humidity_pct=55.0, windspeed_kmh=12.0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_build_query_row_matches_training_schema(df):
    row = build_query_row(_args())
    X_query = add_engineered_columns(row)[FEATURE_COLUMNS]
    X_train, _ = build_matrices(df.head(5))
    assert list(X_query.columns) == list(X_train.columns)
    assert not X_query.isna().any().any()


def test_build_query_row_rejects_out_of_range_inputs():
    # 999 C is not a value TEMP_MAX_C-normalization can represent in [0, 1].
    with pytest.raises(SystemExit):
        build_query_row(_args(temp_c=999.0))


def test_predict_hour_encodes_year_flag_correctly():
    row_2011 = build_query_row(_args(year=2011))
    row_2012 = build_query_row(_args(year=2012))
    assert row_2011.loc[0, "yr"] == 0
    assert row_2012.loc[0, "yr"] == 1
