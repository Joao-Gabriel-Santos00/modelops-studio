import http from "k6/http";
import { check, sleep } from "k6";

export let options = {
  stages: [
    { duration: "30s", target: 10 },   // ramp to 10 VUs
    { duration: "60s", target: 50 },   // ramp to 50 VUs
    { duration: "120s", target: 100 }, // hold at 100 VUs
    { duration: "30s", target: 10 },   // ramp down
    { duration: "10s", target: 0 }     // stop
  ],
  thresholds: {
    "http_req_duration{type:predict}": ["p(95)<500"], // p95 < 500ms for predict requests
    "http_req_failed": ["rate<0.01"]                 // failed requests < 1%
  },
};

const FEATURE_COUNT = 30;

const TARGET = __ENV.TARGET_HOST;
const ENDPOINT = "/predict";

function makePayload() {
  const features = [];
  for (let i = 0; i < FEATURE_COUNT; i++) {
    features.push(Math.random());
  }
  return JSON.stringify({ features: features });
}

export default function () {
  const url = TARGET + ENDPOINT;
  const params = {
    headers: { "Content-Type": "application/json" },
    tags: { type: "predict" } // tag so thresholds above apply
  };

  let res = http.post(url, makePayload(), params);

  // basic check
  check(res, {
    "status is 200": (r) => r.status === 200,
    "response contains prediction": (r) => {
      try {
        const j = r.json();
        return j && ("prediction" in j || "model_run_id" in j);
      } catch (e) {
        return false;
      }
    }
  });

  // small pause per iteration (optional)
  sleep(0.05);
}
