from datetime import timedelta

import numpy as np

from src import config, storage
from src.models import predict as P
from src.models.train import train_all
from src.processing.features import FEATURES, build_inference_frame, build_training_frame


def test_training_frame_has_no_lookahead(synthetic_inputs):
    for h in (1, 3):
        df = build_training_frame(h, synthetic_inputs)
        assert not df.empty
        assert set(FEATURES) <= set(df.columns)
        assert (df["last_known"] == df["target_date"] - timedelta(days=h)).all()
        assert (df["ref_date"] < df["last_known"]).all()
        # stocks always within quotas
        assert df["red_left"].between(0, config.RED_QUOTA).all()
        assert df["white_left"].between(0, config.WHITE_QUOTA).all()


def test_labels_always_allowed_by_mask(synthetic_inputs):
    df = build_training_frame(1, synthetic_inputs)
    masks = df[["mask_blue", "mask_white", "mask_red"]].to_numpy(bool)
    assert masks[np.arange(len(df)), df["target"].astype(int)].all()


def test_inference_frame_targets_next_unknown_days(synthetic_inputs):
    df = build_inference_frame(synthetic_inputs, horizons=(1, 2))
    last = max(synthetic_inputs.tempo["date"])
    assert df["target_date"].tolist() == [last + timedelta(days=1), last + timedelta(days=2)]


def test_train_predict_store_roundtrip(synthetic_inputs, temp_db, monkeypatch):
    monkeypatch.setattr(config, "HORIZONS", (1, 2))
    meta = train_all(synthetic_inputs, models_dir=config.MODELS_DIR, run_backtest=True)
    assert set(meta["horizons"]) == {"1", "2"}
    bt = meta["horizons"]["1"]["backtest"]["overall"]
    # the synthetic colours are learnable: model must beat "always blue"
    assert bt["model"]["f1_macro"] > bt["baseline_always_blue"]["f1_macro"]

    preds = P.predict_next_days(synthetic_inputs, P.get_bundle(config.MODELS_DIR))
    assert len(preds) == 2
    np.testing.assert_allclose(preds[["p_blue", "p_white", "p_red"]].sum(axis=1), 1.0, atol=1e-6)
    assert P.store_predictions(preds) == 2
    assert len(storage.read_table(storage.predictions)) == 2
