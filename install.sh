#!/usr/bin/env bash
# Usage: ./install.sh HF_DATASET HF_DATASET_DIR MODE

set -Eeuo pipefail

HF_DATASET="${1:?Usage: ./install.sh HF_DATASET HF_DATASET_DIR MODE}"
HF_DATASET_DIR="${2:?Usage: ./install.sh HF_DATASET HF_DATASET_DIR MODE}"
MODE="${3:?Usage: ./install.sh HF_DATASET HF_DATASET_DIR MODE}"
if [[ "${MODE}" != "train" && "${MODE}" != "eval" ]]; then
  echo "MODE must be 'train' or 'eval', got '${MODE}'" >&2
  exit 1
fi

export STABLEWM_HOME=/workspace

SWM_SRC="./stable-worldmodel"
if [ ! -d "${SWM_SRC}/.git" ]; then
  git clone --depth 1 --branch lewm-tdv --single-branch \
    https://github.com/flamma7/stable-worldmodel.git \
    "${SWM_SRC}"
fi

# Install stable-worldmodel in editable mode from fork.
# Use [train,env] extras for MODE=eval (planning/eval scripts), [train] for MODE=train.
SWM_EXTRAS="train"
if [[ "${MODE}" == "eval" ]]; then
  # Make sure gymnasium only set to mujoco
  pip install PyOpenGL PyOpenGL_accelerate
  apt-get update && apt-get install -y libegl1 libopengl0 libgl1 libgles2 libglvnd0
  SWM_EXTRAS="train,env"
fi
pip install -e "${SWM_SRC}[${SWM_EXTRAS}]" transformers==4.57.1 imageio ale-py hf_transfer
pip freeze | grep stable || true
pip freeze | grep transformers || true

wandb login --verify
hf auth whoami

hf download "${HF_DATASET}" --repo-type dataset --local-dir "${STABLEWM_HOME}/datasets/${HF_DATASET_DIR}/"
