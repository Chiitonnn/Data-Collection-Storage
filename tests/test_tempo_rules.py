from datetime import date

import numpy as np
import pandas as pd

from src.processing import tempo_rules as r


def test_season_boundaries():
    assert r.season_label(date(2024, 8, 31)) == "2023-2024"
    assert r.season_label(date(2024, 9, 1)) == "2024-2025"
    assert r.season_start(date(2025, 2, 1)) == date(2024, 9, 1)


def test_red_calendar_rules():
    assert r.red_allowed_by_calendar(date(2025, 1, 15))  # Wednesday in January
    assert not r.red_allowed_by_calendar(date(2025, 1, 18))  # Saturday
    assert not r.red_allowed_by_calendar(date(2025, 4, 2))  # April
    assert not r.red_allowed_by_calendar(date(2024, 12, 25))  # Christmas
    assert not r.red_allowed_by_calendar(date(2025, 11, 11))  # Armistice


def test_mask_sunday_only_blue():
    m = r.allowed_mask(date(2025, 1, 19), red_left=22, white_left=43, consec_red=0, horizon=1)
    assert m.tolist() == [True, False, False]


def test_mask_stock_and_streak():
    wed = date(2025, 1, 15)
    assert r.allowed_mask(wed, 0, 43, 0, 1).tolist() == [True, True, False]
    assert r.allowed_mask(wed, 10, 0, 0, 1).tolist() == [True, False, True]
    assert r.allowed_mask(wed, 10, 10, 5, 1).tolist() == [True, True, False]
    assert r.allowed_mask(wed, 10, 10, 5, 3).tolist() == [True, True, True]  # streak unknown at h>1


def test_apply_mask_renormalises():
    p = r.apply_mask(np.array([[0.2, 0.3, 0.5]]), np.array([[True, True, False]]))
    np.testing.assert_allclose(p, [[0.4, 0.6, 0.0]])


def test_tempo_state_counts_and_streak():
    tempo = pd.DataFrame({
        "date": pd.date_range("2025-01-13", periods=6).date,
        "color": ["RED", "RED", "WHITE", "RED", "BLUE", "BLUE"],
    })
    st = r.tempo_state(tempo)
    assert st.loc[date(2025, 1, 14), "consec_red"] == 2
    assert st.loc[date(2025, 1, 15), "consec_red"] == 0
    assert st.loc[date(2025, 1, 18), "red_used"] == 3
    assert st.loc[date(2025, 1, 18), "white_used"] == 1


def test_stocks_reset_on_new_season():
    st = {"red_used": 22, "white_used": 43}
    assert r.stocks_for_target(date(2025, 9, 2), st, date(2025, 8, 31)) == (22, 43)
    assert r.stocks_for_target(date(2025, 3, 3), st, date(2025, 3, 1)) == (0, 0)
