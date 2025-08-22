# services/model_server/app.py
import os
import logging
import threading
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.middleware.cors import CORSMiddleware


# -------------------------
# Configuration & logging
# -------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
ARTIFACTS_PATH = os.getenv("ARTIFACTS_PATH", os.path.join(os.getcwd(), "artifacts"))
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "ModelOpsStudioModel")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model-server")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# -------------------------
# Prometheus metrics
# -------------------------
# We label metrics by model_run_id so we can compare per-version behavior in Grafana
PRED_COUNTER = Counter("predictions_total", "Total predictions served", ["model_run_id"])
LATENCY = Histogram("prediction_latency_seconds", "Prediction latency seconds", ["model_run_id"])

# -------------------------
# FastAPI app & globals
# -------------------------
app = FastAPI(title="ModelOps Studio - Model Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # development only; replace with your UI origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_lock = threading.Lock()
_current_model = None               # mlflow.pyfunc.PyFuncModel (or similar)
_current_run_id: Optional[str] = None


# -------------------------
# Pydantic request models
# -------------------------
class PredictRequest(BaseModel):
    features: List[float]


# -------------------------
# Utility functions
# -------------------------
def _artifacts_latest_run_file() -> str:
    return os.path.join(ARTIFACTS_PATH, "latest_run_id.txt")


def load_model_from_run(run_id: str) -> None:
    """
    Load model for the given run_id (sets global _current_model and _current_run_id).
    Uses mlflow.pyfunc.load_model with runs:/ URI which will let MLflow + boto3 fetch artifacts from MinIO/S3.
    """
    global _current_model, _current_run_id

    if not run_id:
        raise ValueError("run_id must be provided")

    model_uri_runs = f"runs:/{run_id}/model"
    logger.info("Attempting to load model from %s (tracking URI %s)", model_uri_runs, MLFLOW_TRACKING_URI)

    try:
        model = mlflow.pyfunc.load_model(model_uri_runs)  # will use MLFLOW envs to access artifact store
    except Exception as e:
        logger.exception("Failed to load model from run %s: %s", run_id, e)
        raise

    with _model_lock:
        _current_model = model
        _current_run_id = run_id
    logger.info("Successfully loaded model for run_id=%s (type=%s)", run_id, type(model))


def resolve_initial_model() -> Optional[str]:
    """
    Try to resolve a model run id at startup in this order:
      1) RUN_ID environment variable
      2) artifacts/latest_run_id.txt file (shared volume)
      3) check MLflow Model Registry for a Production stage model (if MODEL_NAME set)
      4) fallback to most recent run found in MLflow
    Returns the resolved run_id or None.
    """
    # 1) env
    env_run = os.getenv("RUN_ID")
    if env_run:
        logger.info("Found RUN_ID in env: %s", env_run)
        return env_run

    # 2) file
    latest_file = _artifacts_latest_run_file()
    if os.path.exists(latest_file):
        try:
            with open(latest_file, "r") as fh:
                rid = fh.read().strip()
                if rid:
                    logger.info("Found latest_run_id.txt -> %s", rid)
                    return rid
        except Exception as ex:
            logger.warning("Failed to read latest_run_id.txt: %s", ex)

    # 3) Model Registry Production stage
    try:
        prod_versions = client.get_latest_versions(name=DEFAULT_MODEL_NAME, stages=["Production"])
        if prod_versions:
            logger.info("Found model in registry Production: run_id=%s version=%s", prod_versions[0].run_id, prod_versions[0].version)
            return prod_versions[0].run_id
    except Exception:
        logger.debug("Model registry lookup failed or no registry configured")

    # 4) fallback to most recent run (best-effort)
    try:
        # search across experiments, ordered by start_time desc
        runs = client.search_runs(experiment_ids=None, filter_string="", max_results=50, order_by=["attributes.start_time DESC"])
        if runs:
            logger.info("Falling back to most recent run: %s", runs[0].info.run_id)
            return runs[0].info.run_id
    except Exception as e:
        logger.warning("Could not fetch recent runs from MLflow: %s", e)

    return None


# -------------------------
# Startup: attempt to auto-load a model (non-blocking)
# -------------------------
@app.on_event("startup")
def startup_load():
    try:
        run_id = resolve_initial_model()
        if run_id:
            try:
                load_model_from_run(run_id)
                logger.info("Auto-loaded model from run %s on startup", run_id)
            except Exception as e:
                logger.warning("Auto-load failed for run %s: %s", run_id, e)
        else:
            logger.info("No initial model resolved at startup; start with no model loaded.")
    except Exception as e:
        logger.exception("Error during startup model resolution: %s", e)


# -------------------------
# API endpoints
# -------------------------
@app.get("/health")
def health():
    """Simple health check."""
    return {"status": "ok", "model_loaded": bool(_current_run_id)}


@app.get("/runs")
def list_runs(limit: int = Query(50, ge=1, le=200)) -> List[Dict[str, Any]]:
    """
    Proxy a list of recent MLflow runs to the UI.
    Robustly iterates experiments and collects up to `limit` runs.
    """
    try:
        # List experiments and ensure we pass a list to search_runs (some MLflow servers don't accept None)
        experiments = client.search_experiments()
        if not experiments:
            logger.warning("No experiments found in MLflow.")
            return []

        exp_ids = [exp.experiment_id for exp in experiments]
        logger.info("Found experiments: %s", exp_ids)

        collected_runs = []
        # Iterate experiments and collect runs (ordered by start_time desc inside each experiment)
        for exp_id in exp_ids:
            try:
                runs_for_exp = client.search_runs(
                    experiment_ids=[exp_id],
                    filter_string="",
                    max_results=limit,
                    order_by=["attributes.start_time DESC"]
                )
            except Exception as e:
                logger.warning("search_runs failed for experiment %s: %s", exp_id, e)
                continue

            if not runs_for_exp:
                continue

            for r in runs_for_exp:
                collected_runs.append(r)
                if len(collected_runs) >= limit:
                    break
            if len(collected_runs) >= limit:
                break

        # Format response
        out = []
        for r in collected_runs:
            info = r.info
            data = r.data
            out.append({
                "run_id": info.run_id,
                "experiment_id": info.experiment_id,
                "start_time": info.start_time,
                "end_time": info.end_time,
                "status": info.status,
                "metrics": getattr(data, "metrics", {}) or {},
                "params": getattr(data, "params", {}) or {},
                "tags": getattr(data, "tags", {}) or {},
                "artifact_uri": info.artifact_uri
            })
        return out

    except Exception as e:
        logger.exception("Failed to list runs (top-level): %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deploy")
def deploy_model(run_id: Optional[str] = None):
    """
    Deploy a model by run_id. Provide run_id as JSON body or query param.
    Example: POST /deploy?run_id=abcd
    """
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required to deploy a model")

    try:
        load_model_from_run(run_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model from run {run_id}: {e}")

    # write the chosen run_id to artifacts/latest_run_id.txt for persistence
    try:
        os.makedirs(ARTIFACTS_PATH, exist_ok=True)
        with open(_artifacts_latest_run_file(), "w") as fh:
            fh.write(run_id)
    except Exception as e:
        logger.warning("Failed to write latest_run_id.txt: %s", e)

    return {"status": "success", "deployed_run_id": run_id}


@app.post("/predict")
def predict(payload: PredictRequest):
    """
    Predict endpoint using the currently loaded model.
    Returns {'prediction': prob_or_value, 'label': label_if_classification, 'model_run_id': ...}
    """
    global _current_model, _current_run_id
    if _current_model is None:
        raise HTTPException(status_code=400, detail="No model deployed. Call /deploy with a run_id first.")

    # build input array
    try:
        features = np.array(payload.features).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid features format: {e}")

    run_label = _current_run_id or "none"

    # instrument metrics per run_id label
    try:
        timer = LATENCY.labels(run_label).time()
        timer.__enter__()  # manual context manager entry
        try:
            # prefer predict_proba if available (probability for class 1)
            if hasattr(_current_model, "predict_proba"):
                proba = _current_model.predict_proba(features)[0]
                # If binary classification, proba shape [p0, p1]
                if proba.shape and len(proba) > 1:
                    prob_value = float(proba[1])
                else:
                    prob_value = float(proba[0])
                label = int(_current_model.predict(features)[0]) if hasattr(_current_model, "predict") else None
                PRED_COUNTER.labels(run_label).inc()
                return {"prediction": prob_value, "label": label, "model_run_id": run_label}
            else:
                # fallback to predict() output
                pred = _current_model.predict(features)
                val = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
                PRED_COUNTER.labels(run_label).inc()
                return {"prediction": val, "model_run_id": run_label}
        finally:
            timer.__exit__(None, None, None)
    except Exception as e:
        logger.exception("Prediction failed for run %s: %s", run_label, e)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/probe_predict")
def probe_predict(payload: PredictRequest):
    if _current_model is None:
        raise HTTPException(status_code=400, detail="No model deployed")
    features = np.array(payload.features).reshape(1, -1)
    # Try to return full arrays for debugging
    resp = {}
    if hasattr(_current_model, "predict_proba"):
        resp["predict_proba"] = _current_model.predict_proba(features).tolist()
    if hasattr(_current_model, "predict"):
        resp["predict"] = _current_model.predict(features).tolist()
    if hasattr(_current_model, "decision_function"):
        try:
            resp["decision_function"] = _current_model.decision_function(features).tolist()
        except Exception:
            resp["decision_function"] = "n/a"
    return {"model_run_id": _current_run_id, "debug": resp}


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
