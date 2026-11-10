"""SQL storage layer (SQLite by default, PostgreSQL via DATABASE_URL).

Tables
------
tempo_days         official Tempo colour per day
consumption_daily  daily aggregates of RTE eCO2mix national data
weather_daily      population-weighted national weather (archive or forecast)
predictions        every forecast issued, to measure real-life performance
"""
from __future__ import annotations

import logging
import math
from datetime import UTC, date, datetime

import pandas as pd
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Engine,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    event,
    func,
    select,
)

from src import config

log = logging.getLogger(__name__)
metadata = MetaData()

tempo_days = Table(
    "tempo_days",
    metadata,
    Column("date", Date, primary_key=True),
    Column("color", String(5), nullable=False),
    Column("season", String(9), nullable=False),
    Column("source", String(32)),
    Column("updated_at", DateTime(timezone=True)),
)

consumption_daily = Table(
    "consumption_daily",
    metadata,
    Column("date", Date, primary_key=True),
    Column("conso_mean", Float),  # MW
    Column("conso_max", Float),
    Column("conso_min", Float),
    Column("eolien_mean", Float),
    Column("solaire_mean", Float),
    Column("prevision_j1_mean", Float),  # RTE day-ahead consumption forecast
    Column("coverage", Float),  # share of expected data points present (0-1)
    Column("source", String(32)),
    Column("updated_at", DateTime(timezone=True)),
)

weather_daily = Table(
    "weather_daily",
    metadata,
    Column("date", Date, primary_key=True),
    Column("temp_mean", Float),
    Column("temp_min", Float),
    Column("temp_max", Float),
    Column("wind100_mean", Float),
    Column("radiation_mean", Float),
    Column("cloud_cover_mean", Float),
    Column("source", String(16)),  # archive | forecast
    Column("updated_at", DateTime(timezone=True)),
)

predictions = Table(
    "predictions",
    metadata,
    Column("target_date", Date, primary_key=True),
    Column("horizon", Integer, primary_key=True),
    Column("last_known_date", Date, nullable=False),
    Column("pred_color", String(5), nullable=False),
    Column("p_blue", Float),
    Column("p_white", Float),
    Column("p_red", Float),
    Column("model_version", String(32)),
    Column("issued_at", DateTime(timezone=True)),
)

TABLES = {t.name: t for t in (tempo_days, consumption_daily, weather_daily, predictions)}

_engine: Engine | None = None
_engine_url: str | None = None


def get_engine() -> Engine:
    """Engine for config.DATABASE_URL (re-created if the URL changed, e.g. in tests)."""
    global _engine, _engine_url
    url = config.DATABASE_URL
    if _engine is None or _engine_url != url:
        if url.startswith("sqlite:///"):
            from pathlib import Path

            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(url, future=True)
        _engine_url = url
        if url.startswith("sqlite"):

            @event.listens_for(_engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _):  # concurrent readers (API, dashboard) + 1 writer
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA busy_timeout=30000")
                cur.close()

    return _engine


def init_db() -> None:
    metadata.create_all(get_engine())


def _clean(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    return value


def upsert(table: Table, df: pd.DataFrame) -> int:
    """Insert or update rows of df (keyed on the table primary key)."""
    if df is None or df.empty:
        return 0
    cols = [c.name for c in table.columns if c.name in df.columns]
    rows = [{c: _clean(v) for c, v in zip(cols, rec)} for rec in df[cols].itertuples(index=False, name=None)]
    if "updated_at" in table.c and "updated_at" not in cols:
        now = datetime.now(UTC)
        for r in rows:
            r["updated_at"] = now
    pk = [c.name for c in table.primary_key.columns]
    engine = get_engine()
    if engine.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert

    with engine.begin() as conn:
        for i in range(0, len(rows), 500):
            chunk = rows[i : i + 500]
            stmt = insert(table).values(chunk)
            update_cols = {c: stmt.excluded[c] for c in chunk[0] if c not in pk}
            stmt = stmt.on_conflict_do_update(index_elements=pk, set_=update_cols) if update_cols else stmt.on_conflict_do_nothing()
            conn.execute(stmt)
    return len(rows)


def read_table(table: Table, where=None) -> pd.DataFrame:
    stmt = select(table)
    if where is not None:
        stmt = stmt.where(where)
    with get_engine().connect() as conn:
        df = pd.read_sql(stmt, conn)
    for col in table.columns:
        if isinstance(col.type, Date) and col.name in df:
            df[col.name] = pd.to_datetime(df[col.name]).dt.date
    return df


def max_date(table: Table, where=None) -> date | None:
    stmt = select(func.max(table.c.date))
    if where is not None:
        stmt = stmt.where(where)
    with get_engine().connect() as conn:
        value = conn.execute(stmt).scalar()
    if isinstance(value, str):  # SQLite may return ISO strings
        value = date.fromisoformat(value[:10])
    if isinstance(value, datetime):
        value = value.date()
    return value
