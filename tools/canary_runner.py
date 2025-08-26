import argparse
import concurrent.futures
import json
import random
import string
import sys
import time
from typing import List

import requests

# attempt to import canary eval from tools (ensure project root is on sys.path)
try:
    from tools.canary_eval import evaluate as canary_evaluate
except Exception:
    # fallback: try relative import if being executed as module inside package
    from canary_eval import evaluate as canary_evaluate  # type: ignore

DEFAULT_FEATURE_COUNT = 30


def random_features(n: int) -> List[float]:
    # generate reasonable dummy floats for features
    return [random.random() for _ in range(n)]


def send_predict(session: requests.Session, url: str, features: List[float]):
    r = session.post(url + "/predict", json={"features": features}, timeout=5)
    return r.status_code, r.text[:200]


def generate_load(target_url: str, requests_total: int, concurrency: int, feat_count: int):
    target = target_url.rstrip("/")
    print(f"Generating {requests_total} requests to {target} with concurrency={concurrency}")
    successes = 0
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        session = requests.Session()
        futures = []
        for _ in range(requests_total):
            feats = random_features(feat_count)
            futures.append(ex.submit(send_predict, session, target, feats))
        for fut in concurrent.futures.as_completed(futures):
            try:
                status, _ = fut.result()
                if 200 <= status < 300:
                    successes += 1
                else:
                    failures += 1
            except Exception:
                failures += 1
    print(f"Load generation complete. success={successes}, failed={failures}")
    return {"success": successes, "failed": failures}


def deploy_model(model_url: str, run_id: str):
    url = model_url.rstrip("/") + f"/deploy?run_id={run_id}"
    print("Deploying model via:", url)
    r = requests.post(url, timeout=10)
    r.raise_for_status()
    return r.json()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--baseline-run-id", default="none")
    p.add_argument("--model-url", default="http://localhost:8000")
    p.add_argument("--prom-url", default="http://localhost:9090")
    p.add_argument("--requests", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--window", default="2m")
    p.add_argument("--feat-count", type=int, default=DEFAULT_FEATURE_COUNT)
    p.add_argument("--revert", action="store_true", help="If canary fails and baseline provided, revert to baseline")
    return p.parse_args()


def main():
    args = parse_args()
    run_id = args.run_id
    baseline = args.baseline_run_id
    model_url = args.model_url
    prom_url = args.prom_url

    # deploy canary
    try:
        deploy_model(model_url, run_id)
    except Exception as e:
        print("Failed to deploy model:", e)
        sys.exit(2)

    # quick sanity wait
    time.sleep(2)

    # generate load
    gen = generate_load(model_url, args.requests, args.concurrency, args.feat_count)

    # wait a bit longer to allow Prometheus scrape window (short window)
    print("Waiting for metrics to be scraped (sleep 10s)...")
    time.sleep(10)

    # compute labels used by metrics (truncate run_id)
    canary_label = str(run_id)[:8]
    baseline_label = "none" if baseline in ("none", "", None) else str(baseline)[:8]

    print(f"Running canary_eval: baseline={baseline_label}, canary={canary_label}, window={args.window}")
    report = canary_evaluate(prom_url, baseline_label, canary_label, window=args.window)
    print("CANARY REPORT:")
    print(json.dumps(report, indent=2))

    decision = report.get("decision", "FAIL")
    if decision == "FAIL":
        print("Canary evaluation FAILED.")
        if args.revert and baseline not in ("none", "", None):
            print(f"Reverting to baseline run {baseline}...")
            try:
                deploy_model(model_url, baseline)
                print("Revert succeeded.")
            except Exception as e:
                print("Failed to revert:", e)
        else:
            print("No revert performed (either revert disabled or no baseline provided).")
    else:
        print("Canary PASS. Consider promoting this model to production stage in MLflow Model Registry.")
    # return non-zero on fail for automation
    sys.exit(0 if decision == "PASS" else 3)


if __name__ == "__main__":
    main()
