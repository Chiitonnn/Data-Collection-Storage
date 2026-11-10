"""Training + honest evaluation of the Tempo colour models (one per horizon).

Evaluation = walk-forward backtest by season: for each season S, the model is
trained on all seasons before S only and evaluated on S. Results are compared
with two baselines (always BLUE, and a temperature/stock heuristic).
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight

from src import config
from src.processing.features import FEATURES, Inputs, build_training_frame
from src.processing.tempo_rules import apply_mask

log = logging.getLogger(__name__)
MASK_COLS = ["mask_blue", "mask_white", "mask_red"]
MIN_TRAIN_SEASONS = 2


def make_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_jobs=4,
        random_state=42,
    )


def sample_weights(y: pd.Series) -> np.ndarray:
    # sqrt of "balanced" weights: rare colours matter more, without over-predicting them
    return np.sqrt(compute_sample_weight("balanced", y))


def fit(df: pd.DataFrame) -> xgb.XGBClassifier:
    model = make_model()
    y = df["target"].astype(int)
    model.fit(df[FEATURES], y, sample_weight=sample_weights(y))
    return model


def predict_proba(model: xgb.XGBClassifier, df: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(df[FEATURES])
    return apply_mask(raw, df[MASK_COLS].to_numpy(dtype=bool))


def heuristic_baseline(df: pd.DataFrame) -> np.ndarray:
    """Cold weekday + stock left -> RED / WHITE, else BLUE (rules respected)."""
    pred = np.zeros(len(df), dtype=int)
    white = df["mask_white"].to_numpy(bool) & (df["temp_mean"] < 5).to_numpy()
    red = df["mask_red"].to_numpy(bool) & (df["temp_mean"] < 2).to_numpy()
    pred[white] = 1
    pred[red] = 2
    return pred


def scores(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)), 4),
        **{f"precision_{c.lower()}": round(float(p[i]), 4) for i, c in enumerate(config.COLORS)},
        **{f"recall_{c.lower()}": round(float(r[i]), 4) for i, c in enumerate(config.COLORS)},
        **{f"f1_{c.lower()}": round(float(f[i]), 4) for i, c in enumerate(config.COLORS)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "n": int(len(y_true)),
    }


def complete_seasons(df: pd.DataFrame) -> list[str]:
    """Seasons with (almost) every day labelled."""
    counts = df.groupby("season").size()
    return sorted(counts[counts >= 360].index)


def backtest(df: pd.DataFrame) -> dict:
    seasons = complete_seasons(df)
    per_season, y_all, model_all, blue_all, heur_all = {}, [], [], [], []
    for i, season in enumerate(seasons):
        if i < MIN_TRAIN_SEASONS:
            continue
        train = df[df["season"] < season]
        test = df[df["season"] == season]
        y = test["target"].astype(int).to_numpy()
        pred = predict_proba(fit(train), test).argmax(axis=1)
        heur = heuristic_baseline(test)
        per_season[season] = {"model": scores(y, pred), "heuristic": scores(y, heur)}
        y_all.append(y)
        model_all.append(pred)
        heur_all.append(heur)
        blue_all.append(np.zeros_like(y))
        log.info(
            "  %s  model F1=%.3f (red recall %.2f) | heuristic F1=%.3f",
            season, per_season[season]["model"]["f1_macro"], per_season[season]["model"]["recall_red"],
            per_season[season]["heuristic"]["f1_macro"],
        )
    if not y_all:
        return {}
    y = np.concatenate(y_all)
    return {
        "seasons": list(per_season),
        "overall": {
            "model": scores(y, np.concatenate(model_all)),
            "baseline_always_blue": scores(y, np.concatenate(blue_all)),
            "baseline_heuristic": scores(y, np.concatenate(heur_all)),
        },
        "per_season": per_season,
    }


def train_all(inputs: Inputs | None = None, models_dir=None, run_backtest: bool = True) -> dict:
    inputs = inputs or Inputs.from_db()
    models_dir = models_dir or config.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    version = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    meta = {"version": version, "trained_at": datetime.now(UTC).isoformat(), "features": FEATURES, "horizons": {}}

    for h in config.HORIZONS:
        df = build_training_frame(h, inputs)
        if df["target"].nunique() < 3:
            raise RuntimeError(f"horizon {h}: not enough labelled data to train ({len(df)} rows)")
        log.info("Horizon %d: %d training days (%s -> %s)", h, len(df), df["target_date"].min(), df["target_date"].max())
        bt = backtest(df) if run_backtest else {}
        model = fit(df)
        model.save_model(models_dir / f"tempo_h{h}.json")
        importance = dict(sorted(zip(FEATURES, model.feature_importances_.round(4).tolist()), key=lambda x: -x[1]))
        meta["horizons"][str(h)] = {
            "n_train": int(len(df)),
            "train_start": str(df["target_date"].min()),
            "train_end": str(df["target_date"].max()),
            "class_counts": {c: int((df["target"] == i).sum()) for i, c in enumerate(config.COLORS)},
            "backtest": bt,
            "feature_importance": importance,
        }
        if bt:
            o = bt["overall"]
            log.info(
                "Horizon %d backtest: model F1 %.3f / acc %.3f | heuristic F1 %.3f | always-blue F1 %.3f acc %.3f",
                h, o["model"]["f1_macro"], o["model"]["accuracy"], o["baseline_heuristic"]["f1_macro"],
                o["baseline_always_blue"]["f1_macro"], o["baseline_always_blue"]["accuracy"],
            )

    tmp = models_dir / "metadata.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(models_dir / "metadata.json")  # atomic: API never reads a half-written file
    log.info("Models v%s saved to %s", version, models_dir)
    return meta


if __name__ == "__main__":
    from src.utils.log import setup_logging

    setup_logging()
    train_all()
