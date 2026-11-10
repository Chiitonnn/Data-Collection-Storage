"""Central configuration: paths, data sources, Tempo constants.

Every value can be overridden with an environment variable (or the .env file).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

# --- Paths / storage -------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", DATA_DIR / "models"))
DATABASE_URL = os.getenv("DATABASE_URL") or f"sqlite:///{(DATA_DIR / 'tempo.db').as_posix()}"

TIMEZONE = "Europe/Paris"

# First day ingested. Public Tempo history starts on 2019-09-01; we take one
# extra month so that lag features exist for the first days of the season.
HISTORY_START = os.getenv("HISTORY_START", "2019-08-01")

# --- Tempo business rules (checked against the real 2019-2026 history) -----
RED_QUOTA = 22
WHITE_QUOTA = 43
MAX_CONSECUTIVE_RED = 5
RED_MONTHS = (11, 12, 1, 2, 3)  # red days only from Nov 1st to Mar 31st
COLORS = ("BLUE", "WHITE", "RED")  # index = class id used by the model
COLOR_TO_ID = {c: i for i, c in enumerate(COLORS)}

# Horizon h = number of days after the last officially published colour.
# h=1 is "the next unknown day" (usually tomorrow before ~11h, the day after
# tomorrow once RTE has published).
HORIZONS = tuple(int(h) for h in os.getenv("HORIZONS", "1,2,3,4,5").split(","))

# --- Weather: population-weighted national temperature ---------------------
# (name, latitude, longitude, weight ~ metropolitan area population, millions)
WEATHER_CITIES = [
    ("Paris", 48.8566, 2.3522, 12.3),
    ("Lyon", 45.7640, 4.8357, 2.3),
    ("Marseille", 43.2965, 5.3698, 1.9),
    ("Lille", 50.6292, 3.0573, 1.5),
    ("Toulouse", 43.6047, 1.4442, 1.5),
    ("Bordeaux", 44.8378, -0.5792, 1.4),
    ("Nantes", 47.2184, -1.5536, 1.0),
    ("Nice", 43.7102, 7.2620, 1.0),
    ("Strasbourg", 48.5734, 7.7521, 0.85),
    ("Rennes", 48.1173, -1.6778, 0.75),
]

# --- External sources -------------------------------------------------------
ODRE_BASE_URL = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets"
TEMPO_PUBLIC_URL = "https://www.api-couleur-tempo.fr/api"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
RTE_BASE_URL = "https://digital.iservices.rte-france.com"
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "60"))
USER_AGENT = "edf-tempo-forecast/2.0 (student project)"


def _secret(name: str) -> str | None:
    """Return a credential, or None when it is empty or an obvious placeholder."""
    value = os.getenv(name, "").strip()
    if not value or any(m in value.lower() for m in ("mock", "your_", "changeme", "xxx")):
        return None
    return value


RTE_CLIENT_ID = _secret("RTE_CLIENT_ID")
RTE_CLIENT_SECRET = _secret("RTE_CLIENT_SECRET")
# auto = RTE API when credentials exist, public API otherwise | rte | public
TEMPO_SOURCE = os.getenv("TEMPO_SOURCE", "auto").lower()

# --- Scheduling / model lifecycle ------------------------------------------
# Local (Europe/Paris) times at which the scheduler runs the daily pipeline.
# 11:30 = just after RTE publishes the next day's colour.
SCHEDULE_TIMES = [t.strip() for t in os.getenv("SCHEDULE_TIMES", "07:00,11:30,18:00").split(",") if t.strip()]
RETRAIN_EVERY_DAYS = int(os.getenv("RETRAIN_EVERY_DAYS", "7"))
