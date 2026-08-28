#!/usr/bin/env bash

shutdown="${1:-false}"
model_type="${MODEL_TYPE:-LEWM_TDV}"

if [[ "$(basename "$PWD")" != "stable-worldmodel" ]]; then
  if [[ -d "${PWD}/stable-worldmodel" ]]; then
    cd "${PWD}/stable-worldmodel"
  else
    echo "Error: cwd is not stable-worldmodel and no stable-worldmodel/ directory in ${PWD}" >&2
    exit 1
  fi
fi

: "${OUTPUT_MODEL_NAME:?OUTPUT_MODEL_NAME is not set}"
: "${LEWM_LR:?LEWM_LR is not set}"

case "${model_type}" in
  LEWM_TDV)
    : "${TDV_ALPHA:?TDV_ALPHA is not set}"
    : "${LEWM_LAMBDA:?LEWM_LAMBDA is not set}"
    echo "TDV_ALPHA=${TDV_ALPHA}"
    echo "LEWM_LAMBDA=${LEWM_LAMBDA}"
    python scripts/train/lewm_tdv.py \
      output_model_name="${OUTPUT_MODEL_NAME}" \
      data.dataset.name=galilai-group/ogb_cube_single \
      loss.tdv.weight=${TDV_ALPHA} \
      loss.sigreg.weight=${LEWM_LAMBDA} \
      optimizer.lr=${LEWM_LR} \
      wandb.enabled=true \
      hf.enabled=true \
      data=ogb \
      trainer.max_epochs=10 \
      num_workers=6
    ;;
  LEWM_VISREG)
    : "${VISREG_LAM_SCALE:?VISREG_LAM_SCALE is not set}"
    : "${VISREG_LAM_SHAPE:?VISREG_LAM_SHAPE is not set}"
    : "${VISREG_LAM_CENTER:?VISREG_LAM_CENTER is not set}"
    : "${VISREG_ALPHA:?VISREG_ALPHA is not set}"
    echo "VISREG_LAM_SCALE=${VISREG_LAM_SCALE}"
    echo "VISREG_LAM_SHAPE=${VISREG_LAM_SHAPE}"
    echo "VISREG_LAM_CENTER=${VISREG_LAM_CENTER}"
    echo "VISREG_ALPHA=${VISREG_ALPHA}"
    python scripts/train/lewm_visreg.py \
      output_model_name="${OUTPUT_MODEL_NAME}" \
      data.dataset.name=galilai-group/ogb_cube_single \
      loss.visreg.alpha=${VISREG_ALPHA} \
      loss.visreg.lam_scale=${VISREG_LAM_SCALE} \
      loss.visreg.lam_shape=${VISREG_LAM_SHAPE} \
      loss.visreg.lam_center=${VISREG_LAM_CENTER} \
      optimizer.lr=${LEWM_LR} \
      wandb.enabled=true \
      hf.enabled=true \
      data=ogb \
      trainer.max_epochs=10 \
      num_workers=6
    ;;
  *)
    echo "Unknown MODEL_TYPE=${model_type} (expected LEWM_TDV or LEWM_VISREG)" >&2
    exit 1
    ;;
esac

status=$?

if [[ "${shutdown,,}" == "true" ]]; then
  runpodctl stop pod "$RUNPOD_POD_ID"
fi

exit "$status"
