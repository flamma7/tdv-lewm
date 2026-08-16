# Runpod instructions
Configures stable-wm and wandb 
```
export STABLEWM_HOME=/workspace
export WANDB_API_KEY=XXX
git clone https://github.com/flamma7/tdv-lewm.git
pip install 'stable-worldmodel[train]' imageio ale-py
wandb login --verify
cd tdv-lewm
python lewm-runpod.py wandb.enabled=true wandb.config.entity=flamma7-myself wandb.config.project=lewm-test output_model_name=lewm_dual_gpu subdir=lewm_dual_gpu n_gpus=2 trainer.devices=2 +trainer.strategy=ddp optimizer.lr=1.5e-4 data.dataset.name=galilai-group/lewm-pusht
```

Also consider downloading the dataset to network drive `/workspace` on a cheap instance