# services/trainer/test_model_load.py
import mlflow
import numpy as np
import json
import sys
# Replace with the same run id you used in the server
RUN_ID = "80f5175f56f647f79acda1e639ee52bf"

mlflow.set_tracking_uri("http://localhost:5000")
model_uri = f"runs:/{RUN_ID}/model"
print("Loading model from:", model_uri)
model = mlflow.sklearn.load_model(model_uri)

print("Model type:", type(model))
# show if model has predict_proba
has_proba = hasattr(model, "predict_proba")
print("Has predict_proba:", has_proba)

# sample input you sent in PowerShell (30 features)
sample = np.array([17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]).reshape(1, -1)

pred = model.predict(sample)
print("predict() ->", pred, type(pred), "as scalar:", float(pred[0]))

if has_proba:
    proba = model.predict_proba(sample)
    print("predict_proba() ->", proba, "prob of class 1:", float(proba[0,1]))
else:
    # try decision_function if available
    if hasattr(model, "decision_function"):
        print("decision_function ->", model.decision_function(sample))
    else:
        print("No predict_proba / decision_function available.")
