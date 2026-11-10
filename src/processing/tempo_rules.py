"""Tempo business rules.

All rules below were checked against the real 2019-2026 history:
- a season runs from Sep 1st to Aug 31st, with 22 RED and 43 WHITE days;
- Sunday is always BLUE;
- RED only Monday-Friday, from Nov 1st to Mar 31st, never on a public holiday;
- never more than 5 RED days in a row;
- WHITE never on a Sunday.
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import holidays
import numpy as np
import pandas as pd

from src import config


def season_start_year(d: date) -> int:
    return d.year if d.month >= 9 else d.year - 1


def season_label(d: date) -> str:
    y = season_start_year(d)
    return f"{y}-{y + 1}"


def season_start(d: date) -> date:
    return date(season_start_year(d), 9, 1)


@lru_cache(maxsize=64)
def _holidays_for_year(year: int) -> frozenset[date]:
    return frozenset(holidays.France(years=year).keys())


def is_holiday(d: date) -> bool:
    return d in _holidays_for_year(d.year)


def red_allowed_by_calendar(d: date) -> bool:
    return d.month in config.RED_MONTHS and d.weekday() < 5 and not is_holiday(d)


def white_allowed_by_calendar(d: date) -> bool:
    return d.weekday() != 6


@lru_cache(maxsize=4096)
def red_eligible_days_left(d: date) -> int:
    """Number of red-eligible days from d (included) to the end of the red period."""
    end = date(season_start_year(d) + 1, 3, 31)
    n, cur = 0, d
    while cur <= end:
        n += red_allowed_by_calendar(cur)
        cur += timedelta(days=1)
    return n


@lru_cache(maxsize=4096)
def white_eligible_days_left(d: date) -> int:
    end = date(season_start_year(d) + 1, 8, 31)
    total = (end - d).days + 1
    sundays = sum(1 for i in range(total) if (d + timedelta(days=i)).weekday() == 6)
    return total - sundays


def tempo_state(tempo: pd.DataFrame) -> pd.DataFrame:
    """State of the Tempo season at the END of each known day K.

    Returns a frame indexed by date with: season, red_used, white_used,
    consec_red (red streak ending at K) and last_color_id (colour of K).
    """
    t = tempo[["date", "color"]].copy()
    t["date"] = pd.to_datetime(t["date"])
    t = t.set_index("date").sort_index()
    t = t.reindex(pd.date_range(t.index.min(), t.index.max(), freq="D"))
    season = [season_label(d.date()) for d in t.index]
    is_red = (t["color"] == "RED").astype(int)
    is_white = (t["color"] == "WHITE").astype(int)
    out = pd.DataFrame(index=t.index)
    out["season"] = season
    out["red_used"] = is_red.groupby(season).cumsum()
    out["white_used"] = is_white.groupby(season).cumsum()
    streak_id = (is_red != is_red.shift()).cumsum()
    out["consec_red"] = is_red.groupby(streak_id).cumsum() * is_red
    out["last_color_id"] = t["color"].map(config.COLOR_TO_ID)
    out.index = out.index.date
    return out


def stocks_for_target(target: date, state_row: pd.Series | dict, last_known: date) -> tuple[int, int]:
    """Remaining (red, white) stock for the target day, knowing colours up to last_known."""
    if season_label(target) != season_label(last_known):
        return config.RED_QUOTA, config.WHITE_QUOTA
    return (
        max(config.RED_QUOTA - int(state_row["red_used"]), 0),
        max(config.WHITE_QUOTA - int(state_row["white_used"]), 0),
    )


def allowed_mask(target: date, red_left: int, white_left: int, consec_red: int, horizon: int) -> np.ndarray:
    """Boolean mask [BLUE, WHITE, RED] of the colours that are still possible."""
    if target.weekday() == 6:
        return np.array([True, False, False])
    white_ok = white_allowed_by_calendar(target) and white_left > 0
    red_ok = red_allowed_by_calendar(target) and red_left > 0
    if horizon == 1 and consec_red >= config.MAX_CONSECUTIVE_RED:
        red_ok = False
    return np.array([True, white_ok, red_ok])


def apply_mask(probs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Zero out impossible colours and renormalise each row."""
    p = np.where(masks, probs, 0.0)
    s = p.sum(axis=1, keepdims=True)
    return np.where(s > 0, p / np.where(s == 0, 1, s), np.array([[1.0, 0.0, 0.0]]))
