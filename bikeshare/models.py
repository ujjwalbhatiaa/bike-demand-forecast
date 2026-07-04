"""The model ladder compared in train.py — from dumb baseline to boosted trees.

Every serious ML comparison needs a baseline: if a fancy model can't beat
"predict the training mean", it has learned nothing.
"""

from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def get_models() -> dict[str, object]:
    """Name -> unfitted estimator, ordered from simplest to strongest."""
    return {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Ridge regression": Pipeline(
            [("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
        ),
    }
