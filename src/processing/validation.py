"""Lightweight data-quality checks run before anything is written to the store."""
from __future__ import annotations

import logging

import pandas as pd

from src import config

log = logging.getLogger(__name__)


class DataValidationError(ValueError):
    pass


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = set(cols) - set(df.columns)
    if missing:
        raise DataValidationError(f"{name}: missing columns {sorted(missing)}")
    if df["date"].duplicated().any():
        raise DataValidationError(f"{name}: duplicated dates")


def validate_tempo(df: pd.DataFrame) -> None:
    if df.empty:
        raise DataValidationError("tempo: no data received")
    _require(df, ["date", "color", "season"], "tempo")
    bad = set(df["color"]) - set(config.COLORS)
    if bad:
        raise DataValidationError(f"tempo: unknown colours {bad}")
    counts = df.groupby("season")["color"].value_counts().unstack(fill_value=0)
    for col, quota in (("RED", config.RED_QUOTA), ("WHITE", config.WHITE_QUOTA)):
        if col in counts and (counts[col] > quota).any():
            log.warning("tempo: %s quota exceeded in %s", col, counts.index[counts[col] > quota].tolist())


def validate_consumption(df: pd.DataFrame) -> None:
    if df.empty:
        raise DataValidationError("consumption: no data received")
    _require(df, ["date", "conso_mean", "coverage"], "consumption")
    c = df["conso_mean"].dropna()
    if ((c < 15_000) | (c > 110_000)).any():
        raise DataValidationError("consumption: values outside the plausible 15-110 GW range")


def validate_weather(df: pd.DataFrame) -> None:
    if df.empty:
        return  # archive can legitimately be empty when already up to date
    _require(df, ["date", "temp_mean"], "weather")
    t = df["temp_mean"]
    if ((t < -30) | (t > 45)).any():
        raise DataValidationError("weather: temperature outside plausible range")
