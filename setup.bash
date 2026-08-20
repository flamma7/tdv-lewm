export STABLEWM_HOME=/workspace
HF_DATASET=galilai-group/lewm-pusht
HF_DATASET_DIR=galilai-group--lewm-pusht

pip install 'stable-worldmodel[train]==0.1.1' transformers==4.57.1 imageio ale-py hf_transfer
pip freeze |grep stable
pip freeze |grep transformers

wandb login --verify
hf auth login

hf download ${HF_DATASET} --repo-type dataset --local-dir ${STABLEWM_HOME}/datasets/${HF_DATASET_DIR}/