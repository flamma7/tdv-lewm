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

: "${MODE:?MODE is not set}"
if [[ "${MODE}" != "train" && "${MODE}" != "eval" ]]; then
  echo "MODE must be 'train' or 'eval', got '${MODE}'" >&2
  exit 1
fi
: "${HF_DATASET:?HF_DATASET is not set}"
: "${HF_DATASET_DIR:?HF_DATASET_DIR is not set}"

LOG_FILE=/workspace/startup.log
ERR_FILE=/workspace/startup_error.log
mkdir -p /workspace
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# TODO: set up Vector and Axiom log streaming

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
    echo "MODE=${MODE:-}"
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

echo "MODE=${MODE}"
echo "HF_DATASET=${HF_DATASET}"
echo "HF_DATASET_DIR=${HF_DATASET_DIR}"

echo "Running smoke test..."
python smoke_test.py
echo "Smoke test successful."

echo "Installing environment..."
# timeout runs in a subshell; re-export so later steps still see it
export STABLEWM_HOME=/workspace
echo "export STABLEWM_HOME=${STABLEWM_HOME}" >> /root/.bashrc
echo "export test_install='bash ./install.sh \"\${HF_DATASET}\" \"\${HF_DATASET_DIR}\" \"\${MODE}\"'" >> /root/.bashrc

if [[ "${DRY_RUN:-}" == "1" ]]; then
  echo "DRY_RUN is set; skipping install and waiting indefinitely."
  # Export CMD_0 to .bashrc as a new command if set and in DRY_RUN
  if [[ -n "${CMD_0:-}" ]]; then
    echo "export run_cmd_0='${CMD_0}'" >> /root/.bashrc
  fi
  sleep infinity
else
  timeout --kill-after=15s 10m bash ./install.sh "${HF_DATASET}" "${HF_DATASET_DIR}" "${MODE}"
fi

echo "Starting commands"
cd "$(dirname "${BASH_SOURCE[0]}")/stable-worldmodel"
echo "cwd=$(pwd)"

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

runpodctl stop pod "$RUNPOD_POD_ID"
