#!/usr/bin/env python3
"""Exploratory data analysis — writes figures + a summary to reports/.

Usage:  python eda.py [--data data/hour.csv] [--out reports]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bikeshare.data import load_hourly
from bikeshare.features import add_engineered_columns

WEATHER_LABELS = {1: "Clear", 2: "Mist/Cloudy", 3: "Light rain/snow", 4: "Heavy rain/storm"}


def fig_hourly_profile(df: pd.DataFrame, out: Path) -> None:
    """Average demand by hour, working day vs weekend/holiday — the money plot."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for flag, label, color in [(1, "Working day", "#2563eb"), (0, "Weekend / holiday", "#f59e0b")]:
        profile = df[df["workingday"] == flag].groupby("hr")["cnt"].mean()
        ax.plot(profile.index, profile.values, marker="o", label=label, color=color)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average rentals")
    ax.set_title("Hourly demand: commute peaks on working days, midday hump on weekends")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "hourly_profile.png", dpi=150)
    plt.close(fig)


def fig_weather_effect(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    order = sorted(df["weathersit"].unique())
    data = [df.loc[df["weathersit"] == w, "cnt"] for w in order]
    ax.boxplot(data, tick_labels=[WEATHER_LABELS[w] for w in order], showfliers=False)
    ax.set_ylabel("Rentals per hour")
    ax.set_title("Weather situation vs hourly demand")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "weather_effect.png", dpi=150)
    plt.close(fig)


def fig_rider_segments(df: pd.DataFrame, out: Path) -> None:
    """Casual vs registered rider profiles differ sharply by hour."""
    fig, ax = plt.subplots(figsize=(9, 5))
    by_hr = df.groupby("hr")[["casual", "registered"]].mean()
    ax.plot(by_hr.index, by_hr["registered"], marker="o", label="Registered (commuters)", color="#16a34a")
    ax.plot(by_hr.index, by_hr["casual"], marker="s", label="Casual (leisure)", color="#dc2626")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average rentals")
    ax.set_title("Registered riders drive the rush-hour peaks; casual riders ride midday")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "rider_segments.png", dpi=150)
    plt.close(fig)


def fig_temp_vs_demand(df: pd.DataFrame, out: Path) -> None:
    enriched = add_engineered_columns(df)
    binned = enriched.groupby(enriched["temp_c"].round())["cnt"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(binned.index, binned.values, marker="o", color="#7c3aed")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Average rentals per hour")
    ax.set_title("Demand climbs with temperature, flattening near ~30 °C")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "temp_vs_demand.png", dpi=150)
    plt.close(fig)


def write_summary(df: pd.DataFrame, out: Path) -> None:
    lines = [
        "# EDA summary — UCI Bike Sharing (hourly)",
        "",
        f"- Rows: **{len(df):,}** hourly records, {df['dteday'].min().date()} → {df['dteday'].max().date()}",
        f"- Target `cnt`: mean {df['cnt'].mean():.1f}, median {df['cnt'].median():.0f}, max {df['cnt'].max()}",
        f"- Missing values: {int(df.isna().sum().sum())}",
        f"- 2012 vs 2011 mean demand: {df[df.yr == 1].cnt.mean():.0f} vs {df[df.yr == 0].cnt.mean():.0f} "
        "(the service roughly doubled year-over-year — why `yr` must be a feature)",
        f"- Registered share of all rides: {df.registered.sum() / df.cnt.sum():.0%}",
        "",
        "Figures: `hourly_profile.png`, `weather_effect.png`, `rider_segments.png`, `temp_vs_demand.png`",
    ]
    (out / "EDA-SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/hour.csv")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    out = Path(args.out) / "figures"
    out.mkdir(parents=True, exist_ok=True)

    df = load_hourly(args.data)
    fig_hourly_profile(df, out)
    fig_weather_effect(df, out)
    fig_rider_segments(df, out)
    fig_temp_vs_demand(df, out)
    write_summary(df, Path(args.out))
    print(f"EDA complete — 4 figures in {out}, summary in {args.out}/EDA-SUMMARY.md")


if __name__ == "__main__":
    main()
