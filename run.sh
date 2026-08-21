#!/usr/bin/env bash

shutdown="${1:-false}"

python lewm.py -cn lewm.yaml output_model_name=lewm_lr1.225e-4_bs256_gpu3 data.dataset.name=galilai-group/ogb_cube_single wandb.enabled=true hf.enabled=true defaults.data=ogb
status=$?

if [[ "${shutdown,,}" == "true" ]]; then
  runpodctl pod stop "$RUNPOD_POD_ID"
fi

exit "$status"
