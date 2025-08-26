from __future__ import annotations
import argparse
import json
import math
import sys
import time
from datetime import datetime, timedelta
from typing import Any

import requests


def now_iso():
    return datetime.utcnow().isoformat() + "Z"


def format_ts(dt: datetime):
    return dt.isoformat() + "Z"


def parse_window_to_seconds(s: str) -> int:
    s = str(s).strip()
    if s.endswith("s"):
        return int(s[:-1])
    if s.endswith("m"):
        return int(s[:-1]) * 60
    if s.endswith("h"):
        return int(s[:-1]) * 3600
    try:
        return int(s)
    except Exception:
        return 120


def query_prometheus_instant(prom: str, query: str, timeout: int = 20) -> dict:
    url = prom.rstrip("/") + "/api/v1/query"
    params = {"query": query}
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus instant query not success: {payload}")
    return payload["data"]


def extract_scalar_from_instant(data: dict) -> float:
    """
    Return a float value from prometheus instant result or math.nan if missing.
    """
    try:
        res = data.get("result", [])
        if not res:
            return math.nan
        # take first vector sample
        val = res[0]["value"][1]
        return float(val)
    except Exception:
        return math.nan


def compute_p95(prom: str, model_label: str, latency_bucket_metric: str, window: str) -> float:
    """
    Compute p95 via histogram_quantile over rate(latency_bucket[window]) filtered by model label.
    """
    # e.g. histogram_quantile(0.95, sum(rate(prediction_latency_seconds_bucket{model="abc"}[2m])) by (le))
    if not model_label:
        return math.nan
    q = (
        f'histogram_quantile(0.95, '
        f'sum(rate({latency_bucket_metric}{{model="{model_label}"}}[{window}])) by (le))'
    )
    data = query_prometheus_instant(prom, q)
    return extract_scalar_from_instant(data)


def compute_error_rate(prom: str, model_label: str, error_metric: str, total_metric: str, window: str) -> float:
    """
    Compute error rate = increase(error_counter[window]) / increase(total_counter[window])
    If error metric is missing, treat errors as 0.
    """
    tot_q = f'increase({total_metric}{{model="{model_label}"}}[{window}])'
    err_q = f'increase({error_metric}{{model="{model_label}"}}[{window}])'

    # total
    try:
        tot = extract_scalar_from_instant(query_prometheus_instant(prom, tot_q))
    except Exception:
        tot = math.nan

    # errors - if metric missing, Prom will return empty -> nan; treat as 0.0
    try:
        err = extract_scalar_from_instant(query_prometheus_instant(prom, err_q))
        if math.isnan(err):
            err = 0.0
    except Exception:
        # if Prometheus returns error for this query treat as 0
        err = 0.0

    if math.isnan(tot) or tot == 0:
        return 0.0
    return float(err) / float(tot)


def evaluate(
    prom: str,
    baseline_label: str,
    canary_label: str,
    window: str = "2m",
    err_threshold: float = 0.005,
    p95_mul: float = 1.2,
    latency_bucket_metric: str = "prediction_latency_seconds_bucket",
    total_metric: str = "predictions_total",
    error_metric: str = "prediction_errors_total",
) -> dict:
    """
    Return a report dict and decision PASS/FAIL.
    """
    report: dict[str, Any] = {
        "meta": {
            "prometheus": prom,
            "baseline_label": baseline_label,
            "canary_label": canary_label,
            "window": window,
            "latency_bucket_metric": latency_bucket_metric,
            "total_metric": total_metric,
            "error_metric": error_metric,
            "err_threshold": err_threshold,
            "p95_mul": p95_mul,
            "timestamp": now_iso(),
        }
    }

    # Compute p95 and error rates
    try:
        baseline_p95 = compute_p95(prom, baseline_label, latency_bucket_metric, window) if baseline_label else math.nan
    except Exception as e:
        return {"error": f"failed baseline p95 query: {e}"}

    try:
        canary_p95 = compute_p95(prom, canary_label, latency_bucket_metric, window) if canary_label else math.nan
    except Exception as e:
        return {"error": f"failed canary p95 query: {e}"}

    try:
        baseline_err = compute_error_rate(prom, baseline_label, error_metric, total_metric, window) if baseline_label else 0.0
    except Exception as e:
        baseline_err = 0.0

    try:
        canary_err = compute_error_rate(prom, canary_label, error_metric, total_metric, window) if canary_label else 0.0
    except Exception as e:
        canary_err = 0.0

    report["baseline"] = {"p95": baseline_p95, "err_rate": baseline_err}
    report["canary"] = {"p95": canary_p95, "err_rate": canary_err}

    # Decision logic
    fail_reasons = []

    # check errors
    if canary_err > baseline_err + err_threshold:
        fail_reasons.append(
            f"canary_err {canary_err:.6f} exceeds baseline_err {baseline_err:.6f} + {err_threshold:.6f}"
        )

    # check p95 only if both are numbers
    if not math.isnan(baseline_p95) and not math.isnan(canary_p95):
        if canary_p95 > baseline_p95 * p95_mul:
            fail_reasons.append(f"canary_p95 {canary_p95:.3f}s > baseline_p95 {baseline_p95:.3f}s * {p95_mul}")
    else:
        report.setdefault("warnings", []).append("missing p95 for baseline or canary; histogram data may be insufficient")

    report["fail_reasons"] = fail_reasons
    report["decision"] = "FAIL" if fail_reasons else "PASS"
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prom", default="http://localhost:9090", help="Prometheus base URL")
    p.add_argument("--baseline", required=True, help="Baseline model label (e.g. 'none' or 'abcd1234' truncated)")
    p.add_argument("--canary", required=True, help="Canary model label (e.g. 'efgh5678')")
    p.add_argument("--window", default="2m", help="Prometheus window (eg 2m, 5m)")
    p.add_argument("--err-threshold", type=float, default=0.005, help="Absolute error-rate delta threshold")
    p.add_argument("--p95-mul", type=float, default=1.2, help="Multiplier threshold for p95")
    p.add_argument("--latency-bucket", default="prediction_latency_seconds_bucket", help="Histogram bucket metric name")
    p.add_argument("--total-metric", default="predictions_total", help="Prediction total counter metric name")
    p.add_argument("--error-metric", default="prediction_errors_total", help="Prediction error counter metric name (optional)")
    args = p.parse_args()

    try:
        report = evaluate(
            prom=args.prom,
            baseline_label=args.baseline,
            canary_label=args.canary,
            window=args.window,
            err_threshold=args.err_threshold,
            p95_mul=args.p95_mul,
            latency_bucket_metric=args.latency_bucket,
            total_metric=args.total_metric,
            error_metric=args.error_metric,
        )
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(2)

    print(json.dumps(report, indent=2))
    if report.get("decision") == "PASS":
        sys.exit(0)
    else:
        sys.exit(3)


if __name__ == "__main__":
    main()
