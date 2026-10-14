#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# One-command chaos demo: start a temporary gateway, wait until it is healthy,
# run scripts/chaos_demo.py against it, then always stop the gateway.
#
# Usage: scripts/run_demo.sh [extra chaos_demo.py args]
# Env:   GROUNDWIRE_PORT (default 8000), DEMO_STARTUP_TIMEOUT_S (default 20)
# --------------------------------------------------------------------------- #
set -euo pipefail

PORT="${GROUNDWIRE_PORT:-8000}"
STARTUP_TIMEOUT_S="${DEMO_STARTUP_TIMEOUT_S:-20}"
URL="http://127.0.0.1:${PORT}"
LOG_FILE="$(mktemp -t groundwire-demo.XXXXXX.log)"
GATEWAY_PID=""

log() { printf '[demo] %s\n' "$*" >&2; }

cleanup() {
    if [[ -n "${GATEWAY_PID}" ]] && kill -0 "${GATEWAY_PID}" 2>/dev/null; then
        log "stopping gateway (pid ${GATEWAY_PID})"
        kill -TERM "${GATEWAY_PID}" 2>/dev/null || true
        wait "${GATEWAY_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Refuse to start a second gateway on a busy port; reuse a running one instead.
if curl -fsS "${URL}/healthz" >/dev/null 2>&1; then
    log "a gateway is already running at ${URL}; using it"
    exec python scripts/chaos_demo.py --url "${URL}" "$@"
fi

if ! command -v groundwire >/dev/null 2>&1; then
    log "the 'groundwire' command was not found. Activate your venv and run: pip install -e '.[dev]'"
    exit 2
fi

log "starting gateway on ${URL} (logs: ${LOG_FILE})"
GROUNDWIRE_PORT="${PORT}" GROUNDWIRE_LOG_JSON=true groundwire >"${LOG_FILE}" 2>&1 &
GATEWAY_PID=$!

deadline=$((SECONDS + STARTUP_TIMEOUT_S))
until curl -fsS "${URL}/healthz" >/dev/null 2>&1; do
    if ! kill -0 "${GATEWAY_PID}" 2>/dev/null; then
        log "gateway exited during startup. Last log lines:"
        tail -n 20 "${LOG_FILE}" >&2
        exit 2
    fi
    if (( SECONDS >= deadline )); then
        log "gateway not healthy after ${STARTUP_TIMEOUT_S}s. Last log lines:"
        tail -n 20 "${LOG_FILE}" >&2
        exit 2
    fi
    sleep 0.3
done
log "gateway is healthy"

python scripts/chaos_demo.py --url "${URL}" "$@"
