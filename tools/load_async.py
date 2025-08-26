import argparse
import asyncio
import json
import random
import statistics
import time
from typing import List, Tuple

import httpx


def random_features(n: int) -> List[float]:
    return [random.random() for _ in range(n)]


async def worker(name: int, client: httpx.AsyncClient, queue: asyncio.Queue, results: List[Tuple[bool, float]]):
    while True:
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        url, payload = item
        start = time.perf_counter()
        try:
            r = await client.post(url, json=payload, timeout=10.0)
            elapsed = (time.perf_counter() - start) * 1000.0
            ok = 200 <= r.status_code < 300
            results.append((ok, elapsed))
        except Exception:
            elapsed = (time.perf_counter() - start) * 1000.0
            results.append((False, elapsed))
        finally:
            queue.task_done()


async def run_load(target: str, n_requests: int, concurrency: int, feat_count: int, path: str):
    url = target.rstrip("/") + path
    q: asyncio.Queue = asyncio.Queue()
    for _ in range(n_requests):
        q.put_nowait((url, {"features": random_features(feat_count)}))

    results: List[Tuple[bool, float]] = []
    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(concurrency):
            tasks.append(asyncio.create_task(worker(i, client, q, results)))
        t0 = time.perf_counter()
        await asyncio.gather(*tasks)
        t1 = time.perf_counter()

    total = len(results)
    successes = sum(1 for ok, _ in results if ok)
    failures = total - successes
    latencies = [lat for ok, lat in results if ok]
    mean_ms = statistics.mean(latencies) if latencies else 0.0
    p50 = percentile(latencies, 50) if latencies else 0.0
    p95 = percentile(latencies, 95) if latencies else 0.0
    p99 = percentile(latencies, 99) if latencies else 0.0
    duration = t1 - t0
    throughput = total / duration if duration > 0 else 0.0

    report = {
        "target": target,
        "path": path,
        "requests": total,
        "successes": successes,
        "failures": failures,
        "duration_s": round(duration, 3),
        "throughput_req_s": round(throughput, 2),
        "mean_ms": round(mean_ms, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
    }
    print(json.dumps(report, indent=2))
    return report


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[-1]
    d0 = s[f] * (c - k)
    d1 = s[c] * (k - f)
    return d0 + d1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000", help="Base URL (host:port)")
    p.add_argument("--endpoint", default="/predict", help="Endpoint path (should accept POST JSON)")
    p.add_argument("--requests", type=int, default=1000, help="Total requests to issue")
    p.add_argument("--concurrency", type=int, default=50, help="Concurrent workers")
    p.add_argument("--feat-count", type=int, default=30, help="Number of features in payload")
    return p.parse_args()


def main():
    args = parse_args()
    asyncio.run(run_load(args.url, args.requests, args.concurrency, args.feat_count, args.endpoint))


if __name__ == "__main__":
    main()
