# services/model_server/app.py
import os
import logging
import threading
import uuid
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from prometheus_client import (
Counter,
Histogram,
Gauge,
make_asgi_app,
CONTENT_TYPE_LATEST,
generate_latest,
)
from fastapi.middleware.cors import CORSMiddleware
from services.model_server.model_loader import ModelLoader


# -------------------------
# Configuration & logging
# -------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
ARTIFACTS_PATH = os.getenv("ARTIFACTS_PATH", os.path.join(os.getcwd(), "artifacts"))
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "ModelOpsStudioModel")
EXPECTED_FEATURE_COUNT = int(os.getenv("EXPECTED_FEATURE_COUNT", "30"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("model-server")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# -------------------------
# Prometheus metrics
# -------------------------
def _short_run_label(run_id: Optional[str]) -> str:
    if not run_id:
        return "none"
    return str(run_id)[:8]


PRED_COUNTER = Counter("model_predictions_total", "Total predictions served", ["model"])
PRED_ERRORS = Counter("model_prediction_errors_total", "Prediction errors", ["model", "type"])
IN_FLIGHT = Gauge("model_predictions_in_flight", "Number of in-flight prediction requests", ["model"])
LATENCY = Histogram(
"model_prediction_latency_seconds",
"Prediction latency seconds",
["model"],
buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5, 10),
)

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

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

loader = ModelLoader(mlflow_client=client)

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
            # load in background so server starts fast
            def _bg_load(rid):
                try:
                    loader.load(rid)
                    logger.info("Auto-loaded model from run %s on startup", rid)
                except Exception as e:
                    logger.warning("Auto-load failed for run %s: %s", rid, e)
            t = threading.Thread(target=_bg_load, args=(run_id,), daemon=True)
            t.start()
        else:
            logger.info("No initial model resolved at startup; start with no model loaded.")
    except Exception as e:
        logger.exception("Error during startup model resolution: %s", e)


# -------------------------
# Middleware: simple request id and logging
# -------------------------
@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    rid = request.headers.get("X-Request-Id", str(uuid.uuid4()))
    request.state.request_id = rid
    logger.info("req_start %s %s %s", rid, request.method, request.url.path)
    try:
        resp = await call_next(request)
        logger.info("req_end %s %s %s %s", rid, request.method, request.url.path, resp.status_code)
        resp.headers["X-Request-Id"] = rid
        return resp
    except Exception:
        logger.exception("req_err %s %s %s", rid, request.method, request.url.path)
        raise


# -------------------------
# API endpoints
# -------------------------
@app.get("/live")
def liveness():
    return {"status": "alive"}

@app.get("/ready")
def readiness():
    return {"status": "ready" if loader.loaded() else "not_ready"}

@app.get("/health")
def health():
    """Simple health check."""
    return {"status": "ok", "model_loaded": bool(loader.current_run_id())}

@app.get("/runs")
def list_runs(limit: int = Query(50, ge=1, le=200)) -> List[Dict[str, Any]]:
    """
    Proxy a list of recent MLflow runs to the UI.
    Robustly iterates experiments and collects up to `limit` runs.
    """
    try:
        experiments = client.search_experiments()
        if not experiments:
            logger.warning("No experiments found in MLflow.")
            return []


        exp_ids = [exp.experiment_id for exp in experiments]
        logger.info("Found experiments: %s", exp_ids)


        collected_runs = []
        for exp_id in exp_ids:
            try:
                runs_for_exp = client.search_runs(
                experiment_ids=[exp_id],
                filter_string="",
                max_results=limit,
                order_by=["attributes.start_time DESC"],
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


        out = []
        for r in collected_runs:
            info = r.info
            data = r.data
            out.append(
            {
            "run_id": info.run_id,
            "experiment_id": info.experiment_id,
            "start_time": info.start_time,
            "end_time": info.end_time,
            "status": info.status,
            "metrics": getattr(data, "metrics", {}) or {},
            "params": getattr(data, "params", {}) or {},
            "tags": getattr(data, "tags", {}) or {},
            "artifact_uri": info.artifact_uri,
            }
            )
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
        loader.load(run_id)
    except Exception as e:
        logger.exception("Failed to load model %s: %s", run_id, e)
        raise HTTPException(status_code=500, detail=f"Failed to load model from run {run_id}: {e}")

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
    Always returns a JSON object. Never falls through returning None.
    """
    if not loader.loaded():
        logger.warning("Predict called but no model loaded")
        raise HTTPException(status_code=400, detail="No model deployed. Call /deploy with a run_id first.")

    # Validate and build features array
    try:
        features = np.array(payload.features).reshape(1, -1)
    except Exception as e:
        run_label = _short_run_label(loader.current_run_id())
        PRED_ERRORS.labels(run_label, "validation").inc()
        logger.debug("Invalid features payload: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid features format: {e}")

    run_label = _short_run_label(loader.current_run_id())

    # Validate length
    if features.shape[1] != EXPECTED_FEATURE_COUNT:
        PRED_ERRORS.labels(run_label, "validation").inc()
        logger.info("Feature length mismatch: expected %d got %d", EXPECTED_FEATURE_COUNT, features.shape[1])
        raise HTTPException(status_code=400, detail=f"expected {EXPECTED_FEATURE_COUNT} features, got {features.shape[1]}")

    IN_FLIGHT.labels(run_label).inc()
    response = None
    try:
        with LATENCY.labels(run_label).time():
            # delegate to loader.predict which may raise RuntimeError if no model
            try:
                result = loader.predict(features)
            except RuntimeError as e:
                PRED_ERRORS.labels(run_label, "exception").inc()
                logger.exception("Loader predict runtime error: %s", e)
                raise HTTPException(status_code=500, detail=str(e))

            # Normalize result into a response dict
            try:
                # If result is array-like (np.ndarray or list)
                arr = np.array(result)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    prob_value = float(arr[0, 1])
                    label = None
                    try:
                        # best-effort: attempt to get label if model supports predict()
                        if hasattr(loader._model, "predict"):
                            label = int(loader._model.predict(features)[0])
                    except Exception:
                        label = None
                    response = {"prediction": prob_value, "label": label, "model_run_id": loader.current_run_id()}
                else:
                    # scalar fallback
                    val = float(arr.flatten()[0])
                    response = {"prediction": val, "model_run_id": loader.current_run_id()}
            except Exception as e:
                PRED_ERRORS.labels(run_label, "postprocess").inc()
                logger.exception("Error postprocessing prediction result: %s", e)
                raise HTTPException(status_code=500, detail="postprocessing failure")

            # increment success counter
            PRED_COUNTER.labels(run_label).inc()

    finally:
        IN_FLIGHT.labels(run_label).dec()

    # Log and return the deterministic response
    logger.info("Prediction returned for run=%s: %s", loader.current_run_id(), response)
    return response

    
@app.post("/probe_predict")
def probe_predict(payload: PredictRequest):
    if not loader.loaded():
        raise HTTPException(status_code=400, detail="No model deployed")
    features = np.array(payload.features).reshape(1, -1)
    resp = {}
    try:
        if hasattr(loader._model, "predict_proba"):
            resp["predict_proba"] = loader._model.predict_proba(features).tolist()
    except Exception:
        resp["predict_proba"] = "error"
    try:
        if hasattr(loader._model, "predict"):
            resp["predict"] = loader._model.predict(features).tolist()
    except Exception:
        resp["predict"] = "error"
    try:
        if hasattr(loader._model, "decision_function"):
            resp["decision_function"] = loader._model.decision_function(features).tolist()
    except Exception:
        resp["decision_function"] = "n/a"
        return {"model_run_id": loader.current_run_id(), "debug": resp}

@app.get("/metrics_text")
def metrics_text():
    # Keep /metrics mounted for Prometheus ASGI app; provide a text export endpoint if needed
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.on_event("shutdown")
def shutdown_event():
    try:
        loader.unload()
    except Exception:
        logger.exception("Error unloading model on shutdown")
