"""Population-weighted national weather from Open-Meteo (no key needed).

- archive  : observed history (ERA5 / reanalysis), available with a few days delay
- forecast : recent past days + up to 8 days ahead, used to predict future colours
Hourly values are aggregated to local (Europe/Paris) days, then averaged over
the cities of config.WEATHER_CITIES with their population weights.
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src import config
from src.utils.http import make_session

log = logging.getLogger(__name__)
HOURLY = ["temperature_2m", "wind_speed_100m", "shortwave_radiation", "cloud_cover"]
DAILY_COLS = ["temp_mean", "temp_min", "temp_max", "wind100_mean", "radiation_mean", "cloud_cover_mean"]


def _base_params() -> dict:
    return {
        "latitude": ",".join(str(c[1]) for c in config.WEATHER_CITIES),
        "longitude": ",".join(str(c[2]) for c in config.WEATHER_CITIES),
        "hourly": ",".join(HOURLY),
        "timezone": config.TIMEZONE,
    }


def _city_daily(payload: dict) -> pd.DataFrame:
    h = pd.DataFrame(payload["hourly"])
    h["date"] = pd.to_datetime(h["time"]).dt.date
    g = h.groupby("date")
    daily = pd.DataFrame(
        {
            "temp_mean": g["temperature_2m"].mean(),
            "temp_min": g["temperature_2m"].min(),
            "temp_max": g["temperature_2m"].max(),
            "wind100_mean": g["wind_speed_100m"].mean(),
            "radiation_mean": g["shortwave_radiation"].mean(),
            "cloud_cover_mean": g["cloud_cover"].mean(),
        }
    )
    # A day with missing hours (archive delay, partial forecast) is not trusted
    incomplete = g["temperature_2m"].count() < 20
    daily.loc[incomplete, :] = np.nan
    return daily


def _national(payloads: list[dict]) -> pd.DataFrame:
    weights = np.array([c[3] for c in config.WEATHER_CITIES], dtype=float)
    cities = [_city_daily(p) for p in payloads]
    index = cities[0].index
    out = {}
    for col in DAILY_COLS:
        values = np.column_stack([c[col].reindex(index).to_numpy(dtype=float) for c in cities])
        mask = ~np.isnan(values)
        w = np.where(mask, weights, 0.0)
        with np.errstate(invalid="ignore"):
            out[col] = np.where(w.sum(axis=1) >= 0.7 * weights.sum(), np.nansum(values * w, axis=1) / w.sum(axis=1), np.nan)
    df = pd.DataFrame(out, index=index).reset_index().rename(columns={"index": "date"})
    return df.dropna(subset=["temp_mean"])


def _get(url: str, params: dict) -> list[dict]:
    resp = make_session().get(url, params=params, timeout=config.HTTP_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    return payload if isinstance(payload, list) else [payload]


def fetch_archive(start: date, end: date | None = None) -> pd.DataFrame:
    end = end or date.today() - timedelta(days=1)
    frames = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=365), end)
        params = _base_params() | {"start_date": cursor.isoformat(), "end_date": chunk_end.isoformat()}
        frames.append(_national(_get(config.OPEN_METEO_ARCHIVE_URL, params)))
        log.info("Weather archive %s -> %s", cursor, chunk_end)
        cursor = chunk_end + timedelta(days=1)
        if cursor <= end:
            time.sleep(8)  # stay under Open-Meteo per-minute quota
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["source"] = "archive"
    return df


def fetch_forecast(past_days: int = 14, forecast_days: int = 8) -> pd.DataFrame:
    params = _base_params() | {"past_days": past_days, "forecast_days": forecast_days}
    df = _national(_get(config.OPEN_METEO_FORECAST_URL, params))
    df["source"] = "forecast"
    log.info("Weather forecast: %s -> %s", df["date"].min(), df["date"].max())
    return df
