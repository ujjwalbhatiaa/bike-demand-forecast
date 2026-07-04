#!/usr/bin/env python3
"""Train and compare the model ladder; write metrics, figures and a report.

Usage:  python train.py [--data data/hour.csv] [--out reports] [--test-fraction 0.15]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bikeshare.data import load_hourly
from bikeshare.evaluate import regression_metrics, time_split
from bikeshare.features import FEATURE_COLUMNS, build_matrices
from bikeshare.models import get_models


def run_comparison(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, dict]:
    train_df, test_df = time_split(df, test_fraction)
    X_train, y_train = build_matrices(train_df)
    X_test, y_test = build_matrices(test_df)

    rows, fitted = [], {}
    for name, model in get_models().items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        metrics = regression_metrics(y_test, model.predict(X_test))
        rows.append({"model": name, **metrics, "fit_seconds": round(elapsed, 2)})
        fitted[name] = model

    results = pd.DataFrame(rows)
    context = {
        "train_df": train_df, "test_df": test_df,
        "X_test": X_test, "y_test": y_test, "fitted": fitted,
        "train_range": (train_df.dteday.min().date(), train_df.dteday.max().date()),
        "test_range": (test_df.dteday.min().date(), test_df.dteday.max().date()),
    }
    return results, context


def fig_feature_importance(context: dict, out: Path) -> pd.Series:
    rf = context["fitted"]["Random Forest"]
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot.barh(ax=ax, color="#2563eb")
    ax.set_title("Random Forest feature importance")
    ax.set_xlabel("Importance (impurity decrease)")
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=150)
    plt.close(fig)
    return importances


def fig_forecast_week(context: dict, out: Path) -> None:
    """Best model's predictions vs reality over the final 7 test days."""
    test_df, X_test = context["test_df"], context["X_test"]
    best = context["fitted"]["Gradient Boosting"]
    preds = pd.Series(best.predict(X_test), index=test_df.index)

    last_day = test_df["dteday"].max()
    window = test_df[test_df["dteday"] > last_day - pd.Timedelta(days=7)]
    idx = range(len(window))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(idx, window["cnt"].values, label="Actual", color="#111827", linewidth=1.6)
    ax.plot(idx, preds.loc[window.index].values, label="Predicted (Gradient Boosting)",
            color="#dc2626", linewidth=1.4, alpha=0.85)
    ax.set_title(f"Hourly forecast vs actual — final week of test set (ending {last_day.date()})")
    ax.set_xlabel("Hours into the week")
    ax.set_ylabel("Rentals")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "forecast_final_week.png", dpi=150)
    plt.close(fig)


def write_report(results: pd.DataFrame, importances: pd.Series, context: dict, out_dir: Path) -> None:
    tr, te = context["train_range"], context["test_range"]
    best = results.sort_values("rmse").iloc[0]
    baseline = results[results.model == "Mean baseline"].iloc[0]
    top_features = ", ".join(f"`{f}`" for f in importances.sort_values(ascending=False).index[:4])

    lines = [
        "# Model comparison report",
        "",
        f"**Split:** train {tr[0]} → {tr[1]} · test {te[0]} → {te[1]} "
        "(chronological — the model never sees the future).",
        "",
        results.to_markdown(index=False, floatfmt=".2f"),
        "",
        f"**Best model:** {best.model} — RMSE {best.rmse:.1f} rentals/hour "
        f"(vs {baseline.rmse:.1f} for the mean baseline), R² {best.r2:.3f}.",
        "",
        f"**Most informative features:** {top_features}.",
        "",
        "Figures: `feature_importance.png`, `forecast_final_week.png`",
    ]
    (out_dir / "MODEL-REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    results.to_csv(out_dir / "metrics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/hour.csv")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args()

    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_hourly(args.data)
    results, context = run_comparison(df, args.test_fraction)
    importances = fig_feature_importance(context, fig_dir)
    fig_forecast_week(context, fig_dir)
    write_report(results, importances, context, out_dir)

    print(results.to_string(index=False))
    print(f"\nReport written to {out_dir}/MODEL-REPORT.md")


if __name__ == "__main__":
    main()
