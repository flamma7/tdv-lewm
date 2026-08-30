#!/usr/bin/env bash

# LeWM workload startup script.
#
# Assumes the tdv-lewm repository has already been cloned.
# Requires MODE=train|eval plus HF_DATASET and HF_DATASET_DIR.
# Runs the smoke test, install.sh, then each CMD_0, CMD_1, ... until unset.
# Any failure exits non-zero, causing the container to stop.
#
# DRY_RUN:
#   If DRY_RUN is set (export DRY_RUN=1 or similar), this script will skip the
#   main install and command execution logic after performing environment checks.
#   This allows you to manually run portions such as test_install.sh or individual
#   CMD_0 commands for debugging, without executing the full workflow.

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Paths / logging
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE=/workspace/startup.log
ERR_FILE=/workspace/startup_error.log

VECTOR_CONFIG="${SCRIPT_DIR}/vector.yaml"
VECTOR_DATA_DIR=/tmp/vector-data
VECTOR_INTERNAL_LOG=/tmp/vector-internal.log
VECTOR_PID=""
TEE_PID=""
TEE_FIFO=/tmp/startup-tee.fifo

mkdir -p /workspace
mkdir -p "$VECTOR_DATA_DIR"

touch "$LOG_FILE"
touch "$ERR_FILE"
touch "$VECTOR_INTERNAL_LOG"

# Prevent Python from buffering stdout/stderr. This is important for ephemeral
# pods so logs are written immediately rather than potentially being lost.
export PYTHONUNBUFFERED=1

# Capture stdout + stderr from this script and every child process.
# A FIFO + tracked tee is used instead of `exec > >(tee ...)` so we can close
# the pipe and wait for tee to flush before Vector is stopped. Process
# substitution is asynchronous: on ERR, the last error lines often still sit
# in tee's buffer when Vector is killed, so they never reach Axiom.
exec 3>&1 4>&2
rm -f "$TEE_FIFO"
mkfifo "$TEE_FIFO"

if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL tee -a "$LOG_FILE" <"$TEE_FIFO" >&3 2>&4 &
else
  tee -a "$LOG_FILE" <"$TEE_FIFO" >&3 2>&4 &
fi
TEE_PID=$!
exec >"$TEE_FIFO" 2>&1


# ---------------------------------------------------------------------------
# Vector / Axiom
# ---------------------------------------------------------------------------

# Close the FIFO so tee sees EOF, waits for it to exit (flushing the log
# file), then give Vector time to ingest the last lines and ACK to Axiom.
flush_logs() {
  if [[ -n "${TEE_PID:-}" ]]; then
    # Restore the original console fds; this closes the FIFO write end.
    exec 1>&3 2>&4
    wait "$TEE_PID" 2>/dev/null || true
    TEE_PID=""
    rm -f "$TEE_FIFO"
  fi

  sync "$LOG_FILE" 2>/dev/null || sync

  # Vector file source + axiom batch.timeout_secs (0.5). Wait long enough
  # for the last lines to be read, batched, and in-flight to Axiom.
  sleep 2
}

stop_vector() {
  if [[ -n "${VECTOR_PID:-}" ]] && kill -0 "$VECTOR_PID" 2>/dev/null; then
    echo "Flushing Vector logs to Axiom..."
    kill -TERM "$VECTOR_PID"
    wait "$VECTOR_PID" 2>/dev/null || true
    echo "Vector stopped."
  fi
}

write_failure_record() {
  local dest=$1
  {
    echo "===== STARTUP FAILED ${ts} ====="
    echo "timestamp=${ts}"
    echo "exit_code=${ec}"
    echo "failed_command=${failed_cmd}"
    echo "pwd=$(pwd)"
    echo "MODE=${MODE:-}"
    echo "pod_id=${RUNPOD_POD_ID:-}"
  } >> "$dest"
}

on_error() {
  ec=$?

  set +e
  trap - ERR

  failed_cmd=${BASH_COMMAND:-unknown}
  ts=$(date -Is)

  echo "===== STARTUP FAILED ====="
  echo "timestamp=${ts}"
  echo "exit_code=${ec}"
  echo "failed_command=${failed_cmd}"
  echo "pwd=$(pwd)"
  echo "MODE=${MODE:-}"
  echo "pod_id=${RUNPOD_POD_ID:-}"

  # Direct writes: do not depend on the async tee for the record Vector tails.
  write_failure_record "$LOG_FILE"
  write_failure_record "$ERR_FILE"

  flush_logs

  # After tee is closed, append once more so Vector cannot miss the banner.
  write_failure_record "$LOG_FILE"
  write_failure_record "$ERR_FILE"
  sync "$LOG_FILE" 2>/dev/null || sync
  sleep 1

  stop_vector

  echo "Startup failed; stopping pod ${RUNPOD_POD_ID:-unknown}."

  # Uncomment once ready to automatically stop failed pods.
  # runpodctl stop pod "$RUNPOD_POD_ID"

  exit "$ec"
}

trap on_error ERR

# ---------------------------------------------------------------------------
# Install Vector if needed
# ---------------------------------------------------------------------------

if ! command -v vector >/dev/null 2>&1; then
  echo "Vector not found; installing..."

  curl --proto '=https' --tlsv1.2 -sSfL https://sh.vector.dev \
    | bash -s -- -y --prefix /usr/local

  # Ensure the newly installed binary is available.
  export PATH="/usr/local/bin:$PATH"
fi

if ! command -v vector >/dev/null 2>&1; then
  echo "ERROR: Vector installation failed."
  exit 1
fi

echo "Vector available: $(vector --version)"


# ---------------------------------------------------------------------------
# Start Vector
# ---------------------------------------------------------------------------

if [[ ! -f "$VECTOR_CONFIG" ]]; then
  echo "ERROR: Vector config not found: ${VECTOR_CONFIG}"
  exit 1
fi

echo "Starting Vector using ${VECTOR_CONFIG}..."

# Vector's own stdout/stderr must NOT go into startup.log, otherwise Vector
# could ingest its own logs and create a feedback loop.
vector \
  --config "$VECTOR_CONFIG" \
  --require-healthy true \
  --dangerously-allow-env-var-interpolation \
  --graceful-shutdown-limit-secs 20 \
  >"$VECTOR_INTERNAL_LOG" 2>&1 &

VECTOR_PID=$!

# Give Vector a moment to initialize.
sleep 1

if ! kill -0 "$VECTOR_PID" 2>/dev/null; then
  echo "ERROR: Vector failed to start."
  echo "----- Vector internal log -----"
  cat "$VECTOR_INTERNAL_LOG"
  exit 1
fi

echo "Vector started successfully. pid=${VECTOR_PID}"
echo "Streaming logs to Axiom."


# ---------------------------------------------------------------------------
# Validate environment
# ---------------------------------------------------------------------------

: "${MODE:?MODE is not set}"

if [[ "${MODE}" != "train" && "${MODE}" != "eval" ]]; then
  echo "MODE must be 'train' or 'eval', got '${MODE}'" >&2
  exit 1
fi

: "${HF_DATASET:?HF_DATASET is not set}"
: "${HF_DATASET_DIR:?HF_DATASET_DIR is not set}"

echo "MODE=${MODE}"
echo "HF_DATASET=${HF_DATASET}"
echo "HF_DATASET_DIR=${HF_DATASET_DIR}"
echo "RUNPOD_POD_ID=${RUNPOD_POD_ID:-}"
echo "Working directory=$(pwd)"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

echo "Running smoke test..."
python smoke_test.py
echo "Smoke test successful."


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

echo "Installing environment..."

# timeout runs in a subshell; re-export so later steps still see it.
export STABLEWM_HOME=/workspace

echo "export STABLEWM_HOME=${STABLEWM_HOME}" >> /root/.bashrc
echo "export test_install='bash ./install.sh \"\${HF_DATASET}\" \"\${HF_DATASET_DIR}\" \"\${MODE}\"'" >> /root/.bashrc

if [[ "${DRY_RUN:-}" == "1" ]]; then
  echo "DRY_RUN is set; skipping install and waiting indefinitely."

  # Export CMD_0 to .bashrc as a new command if set and in DRY_RUN.
  if [[ -n "${CMD_0:-}" ]]; then
    echo "export run_cmd_0='${CMD_0}'" >> /root/.bashrc
  fi

  sleep infinity
else
  timeout --kill-after=15s 10m \
    bash ./install.sh \
    "${HF_DATASET}" \
    "${HF_DATASET_DIR}" \
    "${MODE}"
fi


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

echo "Starting commands"

cd stable-worldmodel

i=0

while true; do
  varname="CMD_${i}"

  if [[ -z "${!varname+x}" ]]; then
    echo "No ${varname}; command loop complete."
    break
  fi

  echo "Executing ${varname}: ${!varname}"

  eval "${!varname}"

  i=$((i + 1))
done


# ---------------------------------------------------------------------------
# Successful completion
# ---------------------------------------------------------------------------

echo "===== WORKLOAD SUCCEEDED ====="
echo "timestamp=$(date -Is)"
echo "pod_id=${RUNPOD_POD_ID:-}"
echo "All commands completed successfully."

flush_logs

stop_vector

sync

echo "Stopping Runpod pod ${RUNPOD_POD_ID}..."
runpodctl stop pod "$RUNPOD_POD_ID"