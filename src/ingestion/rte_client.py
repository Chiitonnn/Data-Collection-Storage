"""Official RTE API client (optional) for the Tempo calendar.

Requires an application on https://data.rte-france.com subscribed to the
"Tempo Like Supply Contract" API; put its credentials in .env.
"""
from __future__ import annotations

import base64
import logging
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from src import config
from src.utils.http import make_session

log = logging.getLogger(__name__)
PARIS = ZoneInfo(config.TIMEZONE)
MAX_RANGE_DAYS = 365  # the API refuses longer periods


class RTEClient:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.session = make_session()
        self._token: str | None = None
        self._token_expiry = 0.0

    def _get_token(self) -> str:
        if self._token and time.time() < self._token_expiry - 60:
            return self._token
        b64 = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        resp = self.session.post(
            f"{config.RTE_BASE_URL}/token/oauth/",
            headers={"Authorization": f"Basic {b64}", "Content-Type": "application/x-www-form-urlencoded"},
            timeout=config.HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
        self._token = payload["access_token"]
        self._token_expiry = time.time() + int(payload.get("expires_in", 3600))
        return self._token

    def _get(self, endpoint: str, params: dict) -> dict:
        for attempt in range(2):
            resp = self.session.get(
                f"{config.RTE_BASE_URL}{endpoint}",
                headers={"Authorization": f"Bearer {self._get_token()}"},
                params=params,
                timeout=config.HTTP_TIMEOUT,
            )
            if resp.status_code == 401 and attempt == 0:  # token revoked / expired
                self._token = None
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError("unreachable")

    @staticmethod
    def _iso(d: date) -> str:
        """Local midnight with the right UTC offset (+01:00 winter, +02:00 summer)."""
        return datetime(d.year, d.month, d.day, tzinfo=PARIS).isoformat()

    def get_tempo_calendar(self, start: date, end: date) -> pd.DataFrame:
        """Tempo colours for local days in [start, end)."""
        records = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=MAX_RANGE_DAYS), end)
            data = self._get(
                "/open_api/tempo_like_supply_contract/v1/tempo_like_calendars",
                {"start_date": self._iso(cursor), "end_date": self._iso(chunk_end), "fallback_status": "false"},
            )
            for day in data.get("tempo_like_calendars", {}).get("values", []):
                records.append({"date": date.fromisoformat(day["start_date"][:10]), "color": day["value"]})
            cursor = chunk_end
        df = pd.DataFrame(records, columns=["date", "color"])
        df = df[df["color"].isin(config.COLORS)]
        df["source"] = "rte_api"
        return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
