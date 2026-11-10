"""Synthetic but rule-compliant data: tests never hit the network."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src import config, storage
from src.processing import tempo_rules as rules
from src.processing.features import Inputs


def _synthetic(start=date(2019, 9, 1), end=date(2024, 3, 10), seed=0) -> Inputs:
    rng = np.random.default_rng(seed)
    days = pd.date_range(start - timedelta(days=31), end + timedelta(days=7), freq="D").date  # +7: weather forecast
    doy = np.array([d.timetuple().tm_yday for d in days])
    temp = 12 - 8 * np.cos(2 * np.pi * (doy - 20) / 365) + rng.normal(0, 3, len(days))
    weather = pd.DataFrame({"date": days, "temp_mean": temp, "temp_min": temp - 4, "temp_max": temp + 4,
                            "wind100_mean": rng.uniform(5, 40, len(days)), "radiation_mean": rng.uniform(10, 250, len(days)),
                            "cloud_cover_mean": rng.uniform(0, 100, len(days)), "source": "archive"})
    conso = 55000 - 2000 * (temp - 12) + rng.normal(0, 1500, len(days))
    consumption = pd.DataFrame({"date": days, "conso_mean": conso, "conso_max": conso + 8000, "conso_min": conso - 8000,
                                "eolien_mean": 4000.0, "solaire_mean": 1500.0, "prevision_j1_mean": conso,
                                "coverage": 1.0, "source": "test"})
    # colours: coldest eligible days become red/white while stock lasts
    colors, red_used, white_used, streak, season = [], 0, 0, 0, None
    tmap = dict(zip(days, temp))
    for d in pd.date_range(start, end, freq="D").date:
        if rules.season_label(d) != season:
            season, red_used, white_used = rules.season_label(d), 0, 0
        c = "BLUE"
        if rules.red_allowed_by_calendar(d) and tmap[d] < 3 and red_used < config.RED_QUOTA and streak < 5:
            c = "RED"
        elif rules.white_allowed_by_calendar(d) and tmap[d] < 7 and white_used < config.WHITE_QUOTA:
            c = "WHITE"
        red_used += c == "RED"
        white_used += c == "WHITE"
        streak = streak + 1 if c == "RED" else 0
        colors.append((d, c))
    tempo = pd.DataFrame(colors, columns=["date", "color"])
    tempo["season"] = tempo["date"].map(rules.season_label)
    tempo["source"] = "test"
    return Inputs(tempo=tempo, consumption=consumption, weather=weather)


@pytest.fixture(scope="session")
def synthetic_inputs() -> Inputs:
    return _synthetic()


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    storage.init_db()
    yield tmp_path
    storage.get_engine().dispose()
