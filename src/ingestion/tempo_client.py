"""Official Tempo colours.

Default source: api-couleur-tempo.fr (free, no key, full history since 2019-09
and the next day's colour as soon as RTE publishes it).
If RTE credentials are configured, the official RTE API is used instead.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from src import config
from src.ingestion.rte_client import RTEClient
from src.utils.http import make_session

log = logging.getLogger(__name__)
CODE_TO_COLOR = {1: "BLUE", 2: "WHITE", 3: "RED"}  # 0 = not published yet


def fetch_public() -> pd.DataFrame:
    resp = make_session().get(f"{config.TEMPO_PUBLIC_URL}/joursTempo", timeout=config.HTTP_TIMEOUT)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    df = df[df["codeJour"].isin(CODE_TO_COLOR)]
    out = pd.DataFrame({"date": pd.to_datetime(df["dateJour"]).dt.date, "color": df["codeJour"].map(CODE_TO_COLOR)})
    out["source"] = "api-couleur-tempo"
    return out.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def fetch_rte(start: date) -> pd.DataFrame:
    client = RTEClient(config.RTE_CLIENT_ID, config.RTE_CLIENT_SECRET)
    return client.get_tempo_calendar(start, date.today() + timedelta(days=2))


def fetch_tempo(start: date) -> pd.DataFrame:
    source = config.TEMPO_SOURCE
    has_creds = bool(config.RTE_CLIENT_ID and config.RTE_CLIENT_SECRET)
    if source == "rte" or (source == "auto" and has_creds):
        if not has_creds:
            raise RuntimeError("TEMPO_SOURCE=rte but RTE_CLIENT_ID / RTE_CLIENT_SECRET are missing")
        try:
            df = fetch_rte(start)
            log.info("Tempo: %d days from RTE API", len(df))
            return df
        except Exception as exc:  # real data either way: fall back to the public mirror
            if source == "rte":
                raise
            log.warning("RTE API failed (%s), falling back to api-couleur-tempo.fr", exc)
    df = fetch_public()
    log.info("Tempo: %d days from api-couleur-tempo.fr", len(df))
    return df[df["date"] >= start]
