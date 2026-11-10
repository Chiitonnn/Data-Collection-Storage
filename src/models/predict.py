"""Inference: colours of the coming days (official when published, predicted otherwise)."""
from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import xgboost as xgb

from src import config, storage
from src.processing import tempo_rules as rules
from src.processing.features import FEATURES, Inputs, build_inference_frame

log = logging.getLogger(__name__)
WEEKDAYS_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]


class ModelNotAvailable(RuntimeError):
    pass


class ModelBundle:
    """Models for every horizon + metadata, reloaded automatically when retrained."""

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.meta_path = self.models_dir / "metadata.json"
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.meta = meta
        self.version = meta["version"]
        self.features = meta["features"]
        if self.features != FEATURES:
            raise ModelNotAvailable("model was trained with another feature set - retrain it")
        self.models: dict[int, xgb.XGBClassifier] = {}
        for h in meta["horizons"]:
            m = xgb.XGBClassifier()
            m.load_model(self.models_dir / f"tempo_h{h}.json")
            self.models[int(h)] = m
        self.mtime = self.meta_path.stat().st_mtime


_lock = threading.Lock()
_bundle: ModelBundle | None = None


def get_bundle(models_dir: Path | None = None) -> ModelBundle:
    global _bundle
    models_dir = Path(models_dir or config.MODELS_DIR)
    meta_path = models_dir / "metadata.json"
    if not meta_path.exists():
        raise ModelNotAvailable(f"no trained model in {models_dir} - run the pipeline first")
    with _lock:
        if _bundle is None or _bundle.models_dir != models_dir or meta_path.stat().st_mtime != _bundle.mtime:
            _bundle = ModelBundle(models_dir)
            log.info("Loaded models v%s", _bundle.version)
        return _bundle


def predict_next_days(inputs: Inputs | None = None, bundle: ModelBundle | None = None) -> pd.DataFrame:
    """One row per unknown day after the last official colour."""
    inputs = inputs or Inputs.from_db()
    bundle = bundle or get_bundle()
    frame = build_inference_frame(inputs, horizons=tuple(sorted(bundle.models)))
    if frame.empty:
        return frame
    out = []
    for row in frame.to_dict("records"):
        model = bundle.models[row["horizon"]]
        X = pd.DataFrame([row])[FEATURES]
        raw = model.predict_proba(X)
        mask = pd.DataFrame([row])[["mask_blue", "mask_white", "mask_red"]].to_numpy(dtype=bool)
        p = rules.apply_mask(raw, mask)[0]
        out.append(
            {
                "target_date": row["target_date"],
                "horizon": row["horizon"],
                "last_known_date": row["last_known"],
                "pred_color": config.COLORS[int(p.argmax())],
                "p_blue": float(p[0]),
                "p_white": float(p[1]),
                "p_red": float(p[2]),
                "weather_available": bool(pd.notna(row["temp_mean"])),
                "model_version": bundle.version,
            }
        )
    return pd.DataFrame(out)


def store_predictions(pred: pd.DataFrame) -> int:
    if pred.empty:
        return 0
    df = pred[pred["weather_available"]].drop(columns="weather_available").copy()
    df["issued_at"] = datetime.now(UTC)
    return storage.upsert(storage.predictions, df)


def today_paris() -> date:
    return datetime.now(ZoneInfo(config.TIMEZONE)).date()


def season_status(inputs: Inputs, on: date | None = None) -> dict:
    on = on or today_paris()
    label = rules.season_label(on)
    t = inputs.tempo[inputs.tempo["season"] == label]
    red_used = int((t["color"] == "RED").sum())
    white_used = int((t["color"] == "WHITE").sum())
    return {
        "season": label,
        "last_official_date": str(max(inputs.tempo["date"])) if len(inputs.tempo) else None,
        "red_used": red_used,
        "red_left": config.RED_QUOTA - red_used,
        "white_used": white_used,
        "white_left": config.WHITE_QUOTA - white_used,
        "blue_used": int((t["color"] == "BLUE").sum()),
        "red_eligible_days_left": rules.red_eligible_days_left(on + timedelta(days=1)),
    }


def forecast(days: int = 5, inputs: Inputs | None = None) -> dict:
    """Today + the next `days` days: official colour when published, else the model forecast."""
    inputs = inputs or Inputs.from_db()
    today = today_paris()
    official = inputs.tempo.set_index("date")["color"].to_dict()
    try:
        bundle = get_bundle()
        preds = predict_next_days(inputs, bundle).set_index("target_date")
        version = bundle.version
    except ModelNotAvailable:
        preds, version = pd.DataFrame(), None

    result = []
    for i in range(days + 1):
        d = today + timedelta(days=i)
        item = {"date": str(d), "weekday": WEEKDAYS_FR[d.weekday()]}
        if d in official:
            item |= {"status": "official", "color": official[d], "probabilities": None, "horizon": None}
        elif d in preds.index and preds.loc[d, "weather_available"]:
            p = preds.loc[d]
            item |= {
                "status": "predicted",
                "color": p["pred_color"],
                "probabilities": {"BLUE": round(p["p_blue"], 3), "WHITE": round(p["p_white"], 3), "RED": round(p["p_red"], 3)},
                "horizon": int(p["horizon"]),
            }
        else:
            item |= {"status": "unavailable", "color": None, "probabilities": None, "horizon": None}
        result.append(item)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_version": version,
        "season": season_status(inputs, today),
        "days": result,
    }


def prediction_history(season: str | None = None, horizon: int = 1) -> pd.DataFrame:
    """Stored predictions joined with the official colour (real-life track record)."""
    preds = storage.read_table(storage.predictions, storage.predictions.c.horizon == horizon)
    tempo = storage.read_table(storage.tempo_days)
    df = preds.merge(tempo[["date", "color"]], left_on="target_date", right_on="date", how="left")
    df = df.rename(columns={"color": "official_color"}).drop(columns="date")
    df["season"] = df["target_date"].map(rules.season_label)
    df["correct"] = df["official_color"].where(df["official_color"].isna(), df["official_color"] == df["pred_color"])
    if season:
        df = df[df["season"] == season]
    return df.sort_values("target_date")
