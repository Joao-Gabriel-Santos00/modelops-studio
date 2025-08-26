# tests/integration/test_deploy_integration.py
from fastapi.testclient import TestClient
from services.model_server import app as app_module


def test_deploy_endpoint(monkeypatch):
    client = TestClient(app_module.app)

    def fake_load(run_id, *args, **kwargs):
        app_module.loader._model = type(
        "M",
        (),
        {
        "predict_proba": lambda self, x: [[0.3, 0.7]],
        "predict": lambda self, x: [1],
        },
        )()
        app_module.loader._run_id = run_id


        monkeypatch.setattr(app_module.loader, "load", fake_load)
        res = client.post("/deploy?run_id=fake_run")
        assert res.status_code == 200
        assert res.json()["deployed_run_id"] == "fake_run"


        # Now prediction should succeed
        r = client.post("/predict", json={"features": [0.1] * int(app_module.EXPECTED_FEATURE_COUNT)})
        assert r.status_code == 200