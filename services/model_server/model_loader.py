# services/model_server/model_loader.py
import logging
import threading
import mlflow
import time
from typing import Optional

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, mlflow_client=None):
        self._lock = threading.RLock()
        self._model = None
        self._run_id = None
        self._client = mlflow_client
        self._loaded_at = None

    def loaded(self) -> bool:
        return self._model is not None

    def current_run_id(self) -> Optional[str]:
        return self._run_id

    def load(self, run_id: str, max_retries: int = 3, backoff_sec: float = 1.0):
        if not run_id:
            raise ValueError("run_id required")
        uri = f"runs:/{run_id}/model"

        last_exc = None
        for attempt in range(max_retries):
            try:
                logger.info("Loading model %s (attempt %d)", uri, attempt + 1)
                model = mlflow.pyfunc.load_model(uri)
                with self._lock:
                    self._model = model
                    self._run_id = run_id
                    self._loaded_at = time.time()
                logger.info("Loaded model run_id=%s", run_id)
                return
            except Exception as e:
                last_exc = e
                logger.warning("load attempt %d failed for %s: %s", attempt + 1, uri, e)
                time.sleep(backoff_sec * (2 ** attempt))
        raise RuntimeError(f"Failed to load model {run_id}") from last_exc

    def predict(self, X):
        with self._lock:
            if self._model is None:
                raise RuntimeError("No model loaded")
            return self._model.predict_proba(X) if hasattr(self._model, "predict_proba") else self._model.predict(X)

    def unload(self):
        with self._lock:
            self._model = None
            self._run_id = None
            self._loaded_at = None
