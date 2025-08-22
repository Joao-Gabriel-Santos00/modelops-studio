# services/trainer/train.py
import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from mlflow.models.signature import infer_signature

# Config: use env var if set, otherwise default (your local mlflow server)
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", os.path.join(os.getcwd(), "..", "artifacts"))  # defaults to repo/artifacts

mlflow.set_tracking_uri(MLFLOW_URI)

# Ensure artifacts folder exists (this is the shared folder your model server reads)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

print("Using MLflow tracking URI:", MLFLOW_URI)
print("Artifacts folder (shared):", ARTIFACTS_DIR)

# --- Training + logging ---
with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"Starting MLflow Run ID: {run_id}")

    # Load data and train
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)
    clf = RandomForestClassifier(n_estimators=10, random_state=6)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("Model test accuracy:", score)

    # Log basic metric
    mlflow.log_metric("test_accuracy", float(score))

    # Save model to a local temp folder inside the run context (avoid using log_model which triggered the 404)
    local_model_path = "tmp_model"
    # remove old if exists
    if os.path.exists(local_model_path):
        import shutil
        shutil.rmtree(local_model_path)

    # Save model to local path with mlflow.sklearn.save_model
    signature = infer_signature(X_train, clf.predict(X_train))
    mlflow.sklearn.save_model(sk_model=clf, path=local_model_path, signature=signature)
    print(f"Saved model locally to {local_model_path}")

    # Upload the folder as artifacts under artifact_path "model" (this will create runs:/{run_id}/model)
    mlflow.log_artifacts(local_model_path, artifact_path="model")
    print("Uploaded model artifacts to MLflow under artifact_path 'model'")

    # Optional: also write the run_id as a text artifact (handy)
    mlflow.log_text(run_id, "run_id.txt")
    print("Logged run_id as artifact 'run_id.txt'")

    # Also write latest_run_id.txt to your shared artifacts folder so the model server can pick it up
    latest_run_file = os.path.join(ARTIFACTS_DIR, "latest_run_id.txt")
    with open(latest_run_file, "w") as f:
        f.write(run_id)
    print(f"✅ Wrote latest run id to shared artifacts file: {latest_run_file}")

print("Done training and logging.")
