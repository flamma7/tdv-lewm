#!/usr/bin/env bash

shutdown="${1:-false}"
lr="${2:-5e-5}"
alpha="${3:-0.1}"
lamb="${4:-1}"

if [[ "$(basename "$PWD")" != "stable-worldmodel" ]]; then
  if [[ -d "${PWD}/stable-worldmodel" ]]; then
    cd "${PWD}/stable-worldmodel"
  else
    echo "Error: cwd is not stable-worldmodel and no stable-worldmodel/ directory in ${PWD}" >&2
    exit 1
  fi
fi

python scripts/train/lewm_tdv.py \
  output_model_name=a${alpha}_l${lamb}_lr${lr}_gpu2 \
  data.dataset.name=galilai-group/ogb_cube_single \
  loss.tdv.weight=${alpha} \
  loss.sigreg.weight=${lamb} \
  optimizer.lr=${lr} \
  wandb.enabled=true \
  hf.enabled=true \
  data=ogb \
  trainer.max_epochs=10 \
  num_workers=6 \

status=$?

if [[ "${shutdown,,}" == "true" ]]; then
  runpodctl stop pod "$RUNPOD_POD_ID"
fi

exit "$status"
