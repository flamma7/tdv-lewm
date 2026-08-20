

# Standard-192 6 GPUs 
bs 192
lr 1.5e-4
model.predictor.mlp=768
model.predictor.heads=3
num_workers=8
loader.prefetch_factor=1

python lewm.py -cn lewm_standard.yaml output_model_name=standard-192_lr1.225e-4_bs256_gpu3 data.dataset.name=galilai-group/lewm-pusht num_workers=8 wandb.enabled=false hf.enabled=false


# ViT-S 4x 5090's
Let's try bs 256
1.414e-4
num_workers=8
loader.prefetch_factor=1

python lewm.py -cn lewm-vit-s.yaml output_model_name=vit-s_lr1.414e-4_bs256_gpu4 data.dataset.name=galilai-group/lewm-pusht wandb.enabled=false hf.enabled=false

python lewm.py   optimizer.lr=1.414e-4 loader.batch_size=256 n_gpus=4 trainer.devices=4 model.predictor.mlp_dim=768 model.predictor.heads=3 num_workers=8 loader.prefetch_factor=1 

# Regular LeWM

python lewm.py -cn lewm.yaml output_model_name=lewm_lr1.225e-4_bs256_gpu3 data.dataset.name=galilai-group/lewm-pusht wandb.enabled=true hf.enabled=true