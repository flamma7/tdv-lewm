if [ $# -lt 2 ]; then
  echo "Usage: $0 HF_TOKEN WANDB_API_KEY" >&2
  exit 1
fi

export HF_TOKEN="$1"
export WANDB_API_KEY="$2"

export STABLEWM_HOME=/.stablewm # /workspace but let's try in the pod..
HF_DATASET=galilai-group/lewm-pusht
HF_DATASET_DIR=galilai-group--lewm-pusht

pip install 'stable-worldmodel[train]==0.1.1' transformers==4.57.1 imageio ale-py hf_transfer
pip freeze |grep stable
pip freeze |grep transformers

wandb login --verify
hf auth whoami

hf download ${HF_DATASET} --repo-type dataset --local-dir ${STABLEWM_HOME}/datasets/${HF_DATASET_DIR}/
