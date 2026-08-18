import json
import os
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

src = (
    Path(os.environ["STABLEWM_HOME"]).expanduser()
    / "checkpoints"
    / "models--quentinll--lewm-pusht"
)

cfg = OmegaConf.create(
    json.loads((src / "config.json").read_text())
)

print("target:", cfg._target_)

model = hydra.utils.instantiate(cfg)

sd = torch.load(
    src / "weights.pt",
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(sd, strict=True)

print("CHECKPOINT LOAD OK")
print("keys:", len(sd))