#!/usr/bin/env bash

# LeWM workload startup script.
#
# Assumes the tdv-lewm repository has already been cloned.
# Runs the smoke test first, loads the environment, validates required
# variables, and launches the workload.
#
# Any failure exits non-zero, causing the container to stop.

#!/usr/bin/env bash
set -Eeuo pipefail

LOG_FILE=/workspace/lewm_startup.log
ERR_FILE=/workspace/lewm_startup_error.log
mkdir -p /workspace
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

cd "$(dirname "${BASH_SOURCE[0]}")"

on_error() {
  ec=$?
  set +e
  trap - ERR
  failed_cmd=${BASH_COMMAND:-unknown}
  ts=$(date -Is)
  echo "Startup failed (exit $ec) at ${ts}. Command: ${failed_cmd}"
  echo "Flushing logs to ${LOG_FILE} and ${ERR_FILE} before stopping pod ${RUNPOD_POD_ID}..."
  {
    echo "===== STARTUP FAILED ${ts} ====="
    echo "exit_code=${ec}"
    echo "failed_command=${failed_cmd}"
    echo "pwd=$(pwd)"
    echo "ANALYSIS=${ANALYSIS:-}"
    echo "----- log stream -----"
  } >> "$ERR_FILE"
  sync
  sleep 2
  cat "$LOG_FILE" >> "$ERR_FILE"
  sync
  # runpodctl stop pod "$RUNPOD_POD_ID"
  exit "$ec"
}
trap on_error ERR

echo "Running smoke test..."
python smoke_test.py
echo "Smoke test successful."

echo "Loading environment..."
timeout --kill-after=15s 10m bash -ec 'source ./setup.bash'
# timeout runs in a subshell; re-export so later steps still see it
export STABLEWM_HOME=/workspace
echo "export STABLEWM_HOME=${STABLEWM_HOME}" >> /root/.bashrc

if [[ "${ANALYSIS:-}" == "true" ]]; then
  echo "ANALYSIS=true: setup complete, skipping training and sleeping forever."
  exec sleep infinity
fi

echo "Starting tests"

: "${LEWM_LR:?LEWM_LR is not set}"
: "${LEMW_ALPHA:?LEMW_ALPHA is not set}"
: "${LEWM_LAMBDA:?LEWM_LAMBDA is not set}"
: "${OUTPUT_MODEL_NAME:?OUTPUT_MODEL_NAME is not set}"

echo "LEWM_LR=${LEWM_LR}"
echo "LEMW_ALPHA=${LEMW_ALPHA}"
echo "LEWM_LAMBDA=${LEWM_LAMBDA}"
echo "OUTPUT_MODEL_NAME=${OUTPUT_MODEL_NAME}"

exec ./run.sh \
    true \
    "${LEWM_LR}" \
    "${LEMW_ALPHA}" \
    "${LEWM_LAMBDA}"