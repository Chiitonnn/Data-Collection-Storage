from datetime import date

import pandas as pd
from fastapi.testclient import TestClient

from src import storage


def test_upsert_is_idempotent(temp_db):
    df = pd.DataFrame({"date": [date(2025, 1, 1), date(2025, 1, 2)], "color": ["BLUE", "RED"], "season": "2024-2025"})
    storage.upsert(storage.tempo_days, df)
    df.loc[1, "color"] = "WHITE"
    storage.upsert(storage.tempo_days, df)
    out = storage.read_table(storage.tempo_days)
    assert len(out) == 2
    assert out.set_index("date").loc[date(2025, 1, 2), "color"] == "WHITE"
    assert storage.max_date(storage.tempo_days) == date(2025, 1, 2)


def test_api_without_data(temp_db):
    from src.api.main import app

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["model"]["loaded"] is False
        assert client.get("/forecast").status_code == 503
        assert client.get("/forecast?days=99").status_code == 422


def test_api_with_data_and_model(temp_db, synthetic_inputs, monkeypatch):
    from src import config
    from src.api.main import app
    from src.models.train import train_all

    monkeypatch.setattr(config, "HORIZONS", (1, 2))
    storage.upsert(storage.tempo_days, synthetic_inputs.tempo)
    storage.upsert(storage.consumption_daily, synthetic_inputs.consumption)
    storage.upsert(storage.weather_daily, synthetic_inputs.weather)
    train_all(synthetic_inputs, models_dir=config.MODELS_DIR, run_backtest=False)

    with TestClient(app) as client:
        assert client.get("/health").json()["model"]["loaded"] is True
        last = max(synthetic_inputs.tempo["date"])
        assert client.get(f"/tempo/{last}").json()["status"] == "official"
        assert client.get("/history?season=2022-2023").json()["counts"]["RED"] > 0
        assert client.get("/model").status_code == 200
        # predictions for days not published yet must serialise (no NaN in JSON)
        from src.models import predict as P

        P.store_predictions(P.predict_next_days(synthetic_inputs))
        r = client.get("/predictions/track-record")
        assert r.status_code == 200
        assert r.json()["items"][0]["official"] is None
