"""National electricity data (RTE eCO2mix) from the ODRE open-data portal.

No credentials needed. Two datasets are combined:
- eco2mix-national-cons-def : consolidated / definitive data (2012 -> ~last month), 30-min step
- eco2mix-national-tr       : real-time data (recent weeks, incl. RTE day-ahead forecast), 15-min step

Aggregation to daily values is done server side (group_by), so payloads stay tiny.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from src import config
from src.utils.http import make_session

log = logging.getLogger(__name__)

DATASETS = {
    # dataset id: (source label, expected data points per day)
    "eco2mix-national-cons-def": ("odre_def", 48),
    "eco2mix-national-tr": ("odre_tr", 96),
}

SELECT = (
    "date, avg(consommation) as conso_mean, max(consommation) as conso_max, "
    "min(consommation) as conso_min, avg(eolien) as eolien_mean, avg(solaire) as solaire_mean, "
    "avg(prevision_j1) as prevision_j1_mean, count(consommation) as n_points"
)


def fetch_daily_consumption(start: date) -> pd.DataFrame:
    """Daily national consumption / wind / solar / day-ahead forecast since `start` (local dates)."""
    session = make_session()
    frames = []
    for dataset, (source, expected) in DATASETS.items():
        params = {
            "select": SELECT,
            "group_by": "date",
            # date_heure is UTC: take one extra day so the first local day is complete
            "where": f"date_heure >= date'{(start - timedelta(days=1)).isoformat()}'",
            "order_by": "date",
        }
        resp = session.get(f"{config.ODRE_BASE_URL}/{dataset}/exports/json", params=params, timeout=config.HTTP_TIMEOUT)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json())
        if df.empty:
            continue
        df["coverage"] = (df["n_points"] / expected).clip(upper=1.0)
        df["source"] = source
        frames.append(df)
        log.info("ODRE %s: %d days", dataset, len(df))

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["date"] >= start]
    # Same day in both datasets -> keep the consolidated one (listed first)
    df = df.drop_duplicates("date", keep="first").drop(columns="n_points")
    return df.sort_values("date").reset_index(drop=True)
