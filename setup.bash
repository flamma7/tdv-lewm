export STABLEWM_HOME=/workspace
pip install 'stable-worldmodel[train]==0.1.1' transformers==4.57.1 imageio ale-py
pip freeze |grep stable
pip freeze |grep transformers
wandb login --verify
hf auth login