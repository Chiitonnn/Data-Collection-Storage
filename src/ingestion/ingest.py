"""Incremental ingestion into the SQL store (idempotent: safe to re-run)."""
from __future__ import annotations

import logging
from datetime import date, timedelta

from src import config, storage
from src.ingestion import odre_client, tempo_client, weather_client
from src.processing.tempo_rules import season_label
from src.processing.validation import validate_consumption, validate_tempo, validate_weather

log = logging.getLogger(__name__)


def _start(table, refresh_days: int, where=None) -> date:
    """Re-fetch the last `refresh_days` days (late corrections / consolidation)."""
    history_start = date.fromisoformat(config.HISTORY_START)
    last = storage.max_date(table, where)
    if last is None:
        return history_start
    return max(history_start, last - timedelta(days=refresh_days))


def ingest_tempo() -> int:
    start = _start(storage.tempo_days, refresh_days=7)
    df = tempo_client.fetch_tempo(start)
    df["season"] = df["date"].map(season_label)
    validate_tempo(df)
    return storage.upsert(storage.tempo_days, df)


def ingest_consumption() -> int:
    start = _start(storage.consumption_daily, refresh_days=45)
    df = odre_client.fetch_daily_consumption(start)
    validate_consumption(df)
    return storage.upsert(storage.consumption_daily, df)


def ingest_weather() -> int:
    n = 0
    start = _start(storage.weather_daily, refresh_days=10, where=storage.weather_daily.c.source == "archive")
    archive = weather_client.fetch_archive(start)
    validate_weather(archive)
    n += storage.upsert(storage.weather_daily, archive)

    # Forecast never overwrites an observed (archive) day
    forecast = weather_client.fetch_forecast()
    validate_weather(forecast)
    archived = set(storage.read_table(storage.weather_daily, storage.weather_daily.c.source == "archive")["date"])
    forecast = forecast[~forecast["date"].isin(archived)]
    n += storage.upsert(storage.weather_daily, forecast)
    return n
