"""REST API for the Tempo forecast.

Run: uvicorn src.api.main:app --port 8000   (docs on /docs)
The API only reads the SQL store and the model files written by the pipeline;
models are reloaded automatically after each retraining.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src import storage
from src.models import predict as P
from src.processing.features import Inputs
from src.processing.tempo_rules import season_label
from src.utils.log import setup_logging

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    storage.init_db()
    try:
        P.get_bundle()
    except P.ModelNotAvailable as exc:
        log.warning("%s", exc)
    yield


app = FastAPI(
    title="EDF Tempo Forecast API",
    description="Couleur Tempo (BLEU / BLANC / ROUGE) officielle ou prévue pour les prochains jours.",
    version="2.0.0",
    lifespan=lifespan,
)


class DayForecast(BaseModel):
    date: date
    weekday: str
    status: str  # official | predicted | unavailable
    color: str | None
    probabilities: dict[str, float] | None
    horizon: int | None


class SeasonStatus(BaseModel):
    season: str
    last_official_date: date | None
    red_used: int
    red_left: int
    white_used: int
    white_left: int
    blue_used: int
    red_eligible_days_left: int


class Forecast(BaseModel):
    generated_at: str
    model_version: str | None
    season: SeasonStatus
    days: list[DayForecast]


def _inputs() -> Inputs:
    inputs = Inputs.from_db()
    if inputs.tempo.empty:
        raise HTTPException(503, "No data yet: run the pipeline (python -m src.pipeline run)")
    return inputs


@app.get("/health")
def health():
    try:
        bundle = P.get_bundle()
        model = {"loaded": True, "version": bundle.version, "trained_at": bundle.meta["trained_at"]}
    except P.ModelNotAvailable as exc:
        model = {"loaded": False, "detail": str(exc)}
    last = {t.name: str(storage.max_date(t) or "") for t in (storage.tempo_days, storage.consumption_daily, storage.weather_daily)}
    return {"status": "ok" if model["loaded"] else "degraded", "model": model, "last_data_date": last}


@app.get("/forecast", response_model=Forecast)
def get_forecast(days: int = Query(5, ge=1, le=7, description="Nombre de jours après aujourd'hui")):
    return P.forecast(days=days, inputs=_inputs())


@app.get("/tempo/{day}", response_model=DayForecast)
def get_day(day: date):
    inputs = _inputs()
    official = inputs.tempo.set_index("date")["color"]
    if day in official.index:
        return {"date": day, "weekday": P.WEEKDAYS_FR[day.weekday()], "status": "official", "color": official[day],
                "probabilities": None, "horizon": None}
    try:
        preds = P.predict_next_days(inputs)
    except P.ModelNotAvailable as exc:
        raise HTTPException(503, str(exc)) from exc
    row = preds[preds["target_date"] == day]
    if row.empty or not row.iloc[0]["weather_available"]:
        raise HTTPException(404, f"No official colour nor forecast available for {day}")
    r = row.iloc[0]
    return {"date": day, "weekday": P.WEEKDAYS_FR[day.weekday()], "status": "predicted", "color": r["pred_color"],
            "probabilities": {"BLUE": r["p_blue"], "WHITE": r["p_white"], "RED": r["p_red"]}, "horizon": int(r["horizon"])}


@app.get("/season", response_model=SeasonStatus)
def get_season():
    return P.season_status(_inputs())


@app.get("/history")
def get_history(season: str | None = Query(None, pattern=r"^\d{4}-\d{4}$")):
    """Official colours of a season (default: current one)."""
    season = season or season_label(P.today_paris())
    t = storage.read_table(storage.tempo_days, storage.tempo_days.c.season == season)
    if t.empty:
        raise HTTPException(404, f"No data for season {season}")
    t = t.sort_values("date")
    return {"season": season, "counts": t["color"].value_counts().to_dict(),
            "days": [{"date": str(d), "color": c} for d, c in zip(t["date"], t["color"])]}


@app.get("/predictions/track-record")
def track_record(horizon: int = Query(1, ge=1, le=7), season: str | None = Query(None, pattern=r"^\d{4}-\d{4}$")):
    """Predictions actually issued by the pipeline vs. the official colour."""
    df = P.prediction_history(season=season, horizon=horizon)
    done = df[df["official_color"].notna()]
    return {
        "horizon": horizon,
        "n_evaluated": int(len(done)),
        "accuracy": float(done["correct"].astype(float).mean()) if len(done) else None,
        "items": [
            {"target_date": str(r.target_date), "predicted": r.pred_color,
             "official": r.official_color if isinstance(r.official_color, str) else None,
             "p_red": round(r.p_red, 3), "p_white": round(r.p_white, 3)}
            for r in df.itertuples()
        ],
    }


@app.get("/model")
def model_info():
    try:
        meta = P.get_bundle().meta
    except P.ModelNotAvailable as exc:
        raise HTTPException(503, str(exc)) from exc
    return {
        "version": meta["version"],
        "trained_at": meta["trained_at"],
        "features": meta["features"],
        "horizons": {
            h: {k: v for k, v in m.items() if k != "backtest"} | {"backtest_overall": m.get("backtest", {}).get("overall")}
            for h, m in meta["horizons"].items()
        },
    }
