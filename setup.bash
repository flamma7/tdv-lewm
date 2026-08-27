# Run with source setup.bash

export STABLEWM_HOME=/workspace
HF_DATASET=galilai-group/ogb_cube_single
HF_DATASET_DIR=galilai-group--ogb_cube_single
# HF_DATASET=galilai-group/lewm-pusht
# HF_DATASET_DIR=galilai-group--lewm-pusht

SWM_SRC="./stable-worldmodel"
if [ ! -d "${SWM_SRC}/.git" ]; then
  git clone --depth 1 --branch lewm-tdv --single-branch \
    https://github.com/flamma7/stable-worldmodel.git \
    "${SWM_SRC}"
fi

# Editable install from the fork: pulls [train] extras from that pyproject.toml
# (transformers, stable-pretraining, hydra, wandb) plus the 0.1.1 base deps.
# ANALYSIS=true also pulls [env] extras needed for planning/eval scripts.
SWM_EXTRAS="train"
if [[ "${ANALYSIS:-}" == "true" ]]; then
  SWM_EXTRAS="train,env"
fi
pip install -e "${SWM_SRC}[${SWM_EXTRAS}]" transformers==4.57.1 imageio ale-py hf_transfer
pip freeze |grep stable
pip freeze |grep transformers

wandb login --verify
hf auth whoami

hf download ${HF_DATASET} --repo-type dataset --local-dir ${STABLEWM_HOME}/datasets/${HF_DATASET_DIR}/
