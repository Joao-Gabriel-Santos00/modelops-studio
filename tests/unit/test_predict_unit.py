# tests/unit/test_predict_unit.py
import numpy as np
from fastapi.testclient import TestClient
from services.model_server import app as app_module

from services.model_server.model_loader import ModelLoader


# create dummy model object
class DummyModel:
    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])
    def predict(self, X):
        return np.array([1])


def test_predict_happy_path(monkeypatch):
    client = TestClient(app_module.app)

    # inject loader with dummy model
    fake_loader = ModelLoader()
    # bypassing RLock internals for test convenience
    fake_loader._model = DummyModel()
    fake_loader._run_id = "test1234"

    monkeypatch.setattr(app_module, "loader", fake_loader)

    resp = client.post("/predict", json={"features": [0.1] * int(app_module.EXPECTED_FEATURE_COUNT)})
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert body["model_run_id"].startswith("test1234")