#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE=${COMPOSE_FILE:-docker-compose.yml}
# Source environment variables from the .env file if it exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Use the variable from .env, with a fallback default
HEALTH_URL=${MODEL_SERVER_HEALTH_URL:-http://localhost:8000/health}
WAIT_TIMEOUT=${WAIT_TIMEOUT:-240}     # seconds to wait for health (increase if your machine is slow)
SLEEP_SEC=${SLEEP_SEC:-2}
LOG_DIR=infra/logs
LOG_FILE="$LOG_DIR/compose_boot.log"

mkdir -p "$LOG_DIR"
echo "Bringing up infra via ${COMPOSE_FILE} (logs -> ${LOG_FILE})"
# Start services in background
docker compose -f "${COMPOSE_FILE}" up -d --build

# Start a background combined log stream to file (helps debugging if service is slow/crashing)
{
  echo "==== docker compose ps ===="
  docker compose -f "${COMPOSE_FILE}" ps
  echo "==== begin logs (tail -f) ===="
  docker compose -f "${COMPOSE_FILE}" logs --no-color --tail=200 -f &
  LOG_PID=$!
  # let the background log stream run until we kill it after tests
} >> "${LOG_FILE}" 2>&1 &

echo "Waiting for model-server to respond at ${HEALTH_URL} (timeout ${WAIT_TIMEOUT}s)..."

START=$(date +%s)
while true; do
  # use curl to get HTTP status code (and prevent redirection / empty replies)
  HTTP_CODE=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 5 "${HEALTH_URL}" || echo "000")
  if [ "$HTTP_CODE" = "200" ]; then
    echo "service up (HTTP 200) after $(( $(date +%s) - START ))s"
    break
  fi

  # If server accepted connection but sent nothing, curl returns 000 or times out; keep waiting
  echo "waiting for healthy endpoint (status=${HTTP_CODE})..."
  if [ $(( $(date +%s) - START )) -gt "${WAIT_TIMEOUT}" ]; then
    echo "Timed out waiting for health endpoint after ${WAIT_TIMEOUT}s"
    echo "Dumping last 500 lines of logs for debug:"
    docker compose -f "${COMPOSE_FILE}" logs --no-color --tail=500
    # stop background log follower if running
    if [ -n "${LOG_PID:-}" ]; then
      kill "${LOG_PID}" 2>/dev/null || true
    fi
    # exit non-zero so caller knows it failed
    docker compose -f "${COMPOSE_FILE}" down -v
    exit 2
  fi
  sleep "${SLEEP_SEC}"
done

# Run integration tests (use venv python if needed)
echo "Running integration tests..."
python -m pytest -q tests/integration -q
TEST_EXIT=$?

echo "Integration tests finished with exit code $TEST_EXIT"

# cleanup logs follower gracefully
if [ -n "${LOG_PID:-}" ]; then
  sleep 0.5
  kill "${LOG_PID}" 2>/dev/null || true
fi

# Tear down compose
echo "Tearing down infra..."
docker compose -f "${COMPOSE_FILE}" down -v

exit ${TEST_EXIT}
