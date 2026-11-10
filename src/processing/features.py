"""Daily feature engineering.

One row = one target day J, seen from the moment where the last officially
known colour is day K = J - horizon. Only information available at that
moment is used:
- weather of J (observed for training, Open-Meteo forecast at inference);
- consumption of the reference day R = K - 1 (last complete day);
- Tempo season state at the end of K (stocks used, red streak, colour of K);
- calendar of J and the Tempo calendar rules.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src import config, storage
from src.processing import tempo_rules as rules

log = logging.getLogger(__name__)

FEATURES = [
    # weather of target day
    "temp_mean", "temp_min", "temp_max", "temp_3d_mean", "wind100_mean", "radiation_mean", "cloud_cover_mean",
    # consumption of reference day (R = K - 1)
    "conso_ref", "conso_ref_max", "net_conso_ref", "conso_ref_7d", "temp_ref", "temp_delta",
    # calendar of target day
    "dow", "month", "is_holiday", "day_in_season", "red_allowed", "white_allowed",
    # Tempo season state
    "red_left", "white_left", "red_eligible_left", "white_eligible_left", "red_pressure", "white_pressure",
    "last_color_id", "consec_red",
]
MIN_COVERAGE = 0.9


@dataclass
class Inputs:
    tempo: pd.DataFrame
    consumption: pd.DataFrame
    weather: pd.DataFrame

    @classmethod
    def from_db(cls) -> Inputs:
        return cls(
            tempo=storage.read_table(storage.tempo_days),
            consumption=storage.read_table(storage.consumption_daily),
            weather=storage.read_table(storage.weather_daily),
        )


def _assemble(targets: pd.DataFrame, inputs: Inputs) -> pd.DataFrame:
    """targets: columns target_date, horizon, last_known, ref_date."""
    state = rules.tempo_state(inputs.tempo)
    w = inputs.weather.set_index("date").sort_index()
    w_full = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="D").date) if len(w) else w
    temp_3d = w_full["temp_mean"].rolling(3, min_periods=2).mean() if len(w) else pd.Series(dtype=float)

    c = inputs.consumption.set_index("date").sort_index()
    c = c[c["coverage"] >= MIN_COVERAGE]
    c_full = c.reindex(pd.date_range(c.index.min(), c.index.max(), freq="D").date) if len(c) else c
    conso_7d = c_full["conso_mean"].rolling(7, min_periods=5).mean() if len(c) else pd.Series(dtype=float)

    rows = []
    for t in targets.itertuples(index=False):
        J, K, R, h = t.target_date, t.last_known, t.ref_date, t.horizon
        if K not in state.index:
            continue
        st = state.loc[K]
        if pd.isna(st["last_color_id"]):
            continue
        red_left, white_left = rules.stocks_for_target(J, st, K)
        same_season = rules.season_label(J) == rules.season_label(K)
        consec = int(st["consec_red"]) if same_season else 0
        red_elig = rules.red_eligible_days_left(J)
        white_elig = rules.white_eligible_days_left(J)
        wj = w.loc[J] if J in w.index else None
        cr = c.loc[R] if R in c.index else None
        wr = w.loc[R] if R in w.index else None
        temp_mean = float(wj["temp_mean"]) if wj is not None else np.nan
        temp_ref = float(wr["temp_mean"]) if wr is not None else np.nan
        mask = rules.allowed_mask(J, red_left, white_left, consec, h)
        rows.append(
            {
                "target_date": J,
                "horizon": h,
                "last_known": K,
                "ref_date": R,
                "season": rules.season_label(J),
                "temp_mean": temp_mean,
                "temp_min": float(wj["temp_min"]) if wj is not None else np.nan,
                "temp_max": float(wj["temp_max"]) if wj is not None else np.nan,
                "temp_3d_mean": float(temp_3d.get(J, np.nan)),
                "wind100_mean": float(wj["wind100_mean"]) if wj is not None else np.nan,
                "radiation_mean": float(wj["radiation_mean"]) if wj is not None else np.nan,
                "cloud_cover_mean": float(wj["cloud_cover_mean"]) if wj is not None else np.nan,
                "conso_ref": float(cr["conso_mean"]) if cr is not None else np.nan,
                "conso_ref_max": float(cr["conso_max"]) if cr is not None else np.nan,
                "net_conso_ref": float(cr["conso_mean"] - (cr["eolien_mean"] or 0) - (cr["solaire_mean"] or 0))
                if cr is not None
                else np.nan,
                "conso_ref_7d": float(conso_7d.get(R, np.nan)),
                "temp_ref": temp_ref,
                "temp_delta": temp_mean - temp_ref,
                "dow": J.weekday(),
                "month": J.month,
                "is_holiday": int(rules.is_holiday(J)),
                "day_in_season": (J - rules.season_start(J)).days,
                "red_allowed": int(rules.red_allowed_by_calendar(J)),
                "white_allowed": int(rules.white_allowed_by_calendar(J)),
                "red_left": red_left,
                "white_left": white_left,
                "red_eligible_left": red_elig,
                "white_eligible_left": white_elig,
                "red_pressure": red_left / max(red_elig, 1),
                "white_pressure": white_left / max(white_elig, 1),
                "last_color_id": int(st["last_color_id"]),
                "consec_red": consec,
                "mask_blue": mask[0],
                "mask_white": mask[1],
                "mask_red": mask[2],
            }
        )
    return pd.DataFrame(rows)


def build_training_frame(horizon: int, inputs: Inputs) -> pd.DataFrame:
    """All days with a known colour, as they would have been seen `horizon` days before."""
    tempo = inputs.tempo.sort_values("date")
    targets = pd.DataFrame({"target_date": tempo["date"]})
    targets["horizon"] = horizon
    targets["last_known"] = targets["target_date"] - timedelta(days=horizon)
    targets["ref_date"] = targets["last_known"] - timedelta(days=1)
    df = _assemble(targets, inputs)
    if df.empty:
        return df
    labels = tempo.set_index("date")["color"].map(config.COLOR_TO_ID)
    df["target"] = df["target_date"].map(labels)
    # Without weather for the target day the row is useless for training
    return df.dropna(subset=["target", "temp_mean"]).reset_index(drop=True)


def last_complete_consumption_day(inputs: Inputs, before: date) -> date | None:
    c = inputs.consumption
    ok = c[(c["coverage"] >= MIN_COVERAGE) & (c["date"] <= before)]
    return None if ok.empty else max(ok["date"])


def build_inference_frame(inputs: Inputs, horizons=config.HORIZONS) -> pd.DataFrame:
    """Rows for the next unknown days after the last published colour."""
    if inputs.tempo.empty:
        return pd.DataFrame()
    K = max(inputs.tempo["date"])
    R = last_complete_consumption_day(inputs, K - timedelta(days=1)) or (K - timedelta(days=1))
    targets = pd.DataFrame(
        [{"target_date": K + timedelta(days=h), "horizon": h, "last_known": K, "ref_date": R} for h in horizons]
    )
    return _assemble(targets, inputs)


def export_features(path=None) -> pd.DataFrame:
    """Export the horizon-1 training frame to parquet (for the notebook / analysis)."""
    df = build_training_frame(1, Inputs.from_db())
    path = path or config.DATA_DIR / "processed" / "features_h1.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("Features exported to %s (%d rows)", path, len(df))
    return df
