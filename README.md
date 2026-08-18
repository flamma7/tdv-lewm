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

python lewm.py output_model_name=lewm_lr1.5e-4_bs192_gpu6 data.dataset.name=galilai-group/lewm-pusht optimizer.lr=1.5e-4 loader.batch_size=192 wandb.config.entity=flamma7-myself wandb.config.project=lewm-test wandb.enabled=true n_gpus=6 loader.prefetch_factor=2
```

Also consider downloading the dataset to network drive `/workspace` on a cheap instance

I think the issue before was that I was using that dataset instead of lance --> let's try the lance...

# Evaluation Instructions confirmed
1. Install the working package recipe

[requirements-working.txt](https://github.com/user-attachments/files/31180048/requirements-working.txt)

```bash
uv venv --python=3.10 .venv
source .venv/bin/activate
uv pip install -r requirements-working.txt
```

2. Set your `STABLEWM_HOME` variable and create the required directories

```bash
export STABLEWM_HOME=/path/to/your/stablewm-home

mkdir -p "$STABLEWM_HOME/datasets"
mkdir -p "$STABLEWM_HOME/checkpoints/models--quentinll--lewm-pusht"
```

3. Pull the original trained LeWM model directly into SWM's expected checkpoint location. You can try `--revision main` but I'll include this commit hash to assure it works with the frozen Python packages.

```bash
hf download quentinll/lewm-pusht \
    config.json weights.pt \
    --revision 22b330c28c27ead4bfd1888615af1340e3fe9052 \
    --local-dir "$STABLEWM_HOME/checkpoints/models--quentinll--lewm-pusht"
```

4. Confirm your package versions can load LeWM.

[test_load.py](https://github.com/user-attachments/files/31180984/test_load.py)

```bash
python test_load.py
```

5. Now you can load LeWM with swm `0.1.1`. To download and prep the Push-T dataset directly in SWM's expected `datasets` folder, run the following:

```bash
hf download quentinll/lewm-pusht \
    pusht_expert_train.h5.zst \
    --repo-type dataset \
    --local-dir "$STABLEWM_HOME/datasets"

zstd -d \
    "$STABLEWM_HOME/datasets/pusht_expert_train.h5.zst" \
    -o "$STABLEWM_HOME/datasets/pusht_expert_train.h5"
```

Optionally remove the compressed dataset after decompression:

```bash
rm "$STABLEWM_HOME/datasets/pusht_expert_train.h5.zst"
```

6. To evaluate, checkout/clone version `0.1.1` of SWM

```bash
git clone --branch 0.1.1 --depth 1 \
    https://github.com/galilai-group/stable-worldmodel.git \
    stable-worldmodel-0.1.1
```

7. Run the evaluation

```bash
cd stable-worldmodel-0.1.1

python scripts/plan/eval_wm.py \
    policy=quentinll/lewm-pusht
```
