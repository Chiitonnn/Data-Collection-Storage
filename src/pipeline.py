"""Daily pipeline: ingest -> (re)train if needed -> predict -> store predictions.

Usage
-----
python -m src.pipeline run [--retrain]   run once
python -m src.pipeline schedule          run at config.SCHEDULE_TIMES every day (Europe/Paris)
python -m src.pipeline train             retrain + backtest only
python -m src.pipeline predict           print the forecast from data already in the store

Prefect is used for orchestration (retries, run history, UI). Set
DISABLE_PREFECT=1 to run the very same steps as plain Python functions.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from src import config, storage
from src.utils.log import setup_logging

log = logging.getLogger("src.pipeline")

if os.getenv("DISABLE_PREFECT") == "1":

    def task(*_a, **_k):
        return lambda f: f

    def flow(*_a, **_k):
        return lambda f: f

else:
    os.environ.setdefault("PREFECT_LOGGING_EXTRA_LOGGERS", "src")  # show our own logs in Prefect runs
    from prefect import flow, task


@task(name="ingest-tempo", retries=3, retry_delay_seconds=120)
def ingest_tempo_task() -> int:
    from src.ingestion.ingest import ingest_tempo

    return ingest_tempo()


@task(name="ingest-consumption", retries=3, retry_delay_seconds=120)
def ingest_consumption_task() -> int:
    from src.ingestion.ingest import ingest_consumption

    return ingest_consumption()


@task(name="ingest-weather", retries=3, retry_delay_seconds=120)
def ingest_weather_task() -> int:
    from src.ingestion.ingest import ingest_weather

    return ingest_weather()


def _model_age_days() -> float | None:
    meta = config.MODELS_DIR / "metadata.json"
    if not meta.exists():
        return None
    trained_at = datetime.fromisoformat(json.loads(meta.read_text(encoding="utf-8"))["trained_at"])
    return (datetime.now(UTC) - trained_at).total_seconds() / 86400


@task(name="train-models")
def train_task(force: bool = False) -> str:
    from src.models.train import train_all
    from src.processing.features import FEATURES

    age = _model_age_days()
    stale_features = False
    if age is not None:
        meta = json.loads((config.MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
        stale_features = meta.get("features") != FEATURES
    if force or age is None or age >= config.RETRAIN_EVERY_DAYS or stale_features:
        return train_all()["version"]
    log.info("Model is %.1f days old (< %d): no retraining", age, config.RETRAIN_EVERY_DAYS)
    return "unchanged"


@task(name="predict")
def predict_task() -> dict:
    from src.models.predict import forecast, predict_next_days, store_predictions

    n = store_predictions(predict_next_days())
    log.info("%d predictions stored", n)
    return forecast()


@task(name="export-features")
def export_task() -> None:
    from src.processing.features import export_features

    export_features()


@flow(name="tempo-daily-pipeline")
def daily_pipeline(retrain: bool = False) -> dict:
    storage.init_db()
    ingest_tempo_task()
    ingest_consumption_task()
    ingest_weather_task()
    export_task()
    train_task(force=retrain)
    result = predict_task()
    for d in result["days"]:
        log.info("%s %-9s %-10s %s", d["date"], d["weekday"], d["status"], d["color"])
    return result


def _next_run(now: datetime) -> datetime:
    candidates = []
    for day in (0, 1):
        for t in config.SCHEDULE_TIMES:
            hh, mm = (int(x) for x in t.split(":"))
            candidates.append((now + timedelta(days=day)).replace(hour=hh, minute=mm, second=0, microsecond=0))
    return min(c for c in candidates if c > now)


def schedule_forever() -> None:
    tz = ZoneInfo(config.TIMEZONE)
    while True:
        nxt = _next_run(datetime.now(tz))
        log.info("Next run at %s", nxt.isoformat())
        while datetime.now(tz) < nxt:
            time.sleep(min(60, max(1, (nxt - datetime.now(tz)).total_seconds())))
        try:
            daily_pipeline()
        except Exception:  # keep the scheduler alive; Prefect already retried the tasks
            log.exception("Pipeline run failed")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="EDF Tempo forecasting pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--retrain", action="store_true")
    sched = sub.add_parser("schedule")
    sched.add_argument("--run-now", action="store_true", help="run once immediately before scheduling")
    sub.add_parser("train")
    sub.add_parser("predict")
    args = parser.parse_args()

    if args.cmd == "run":
        daily_pipeline(retrain=args.retrain)
    elif args.cmd == "schedule":
        if args.run_now:
            try:
                daily_pipeline()
            except Exception:
                log.exception("Initial run failed")
        schedule_forever()
    elif args.cmd == "train":
        from src.models.train import train_all

        storage.init_db()
        train_all()
    elif args.cmd == "predict":
        from src.models.predict import forecast

        print(json.dumps(forecast(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
