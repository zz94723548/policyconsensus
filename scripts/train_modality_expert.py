"""Train RGB or PCD with an explicit episode split, without starting a simulator."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(version_base=None, config_path=str(ROOT / "modular_policy/config"),
            config_name="train_rgb_expert")
def main(cfg):
    os.chdir(ROOT)
    OmegaConf.resolve(cfg)
    workspace = hydra.utils.instantiate({"_target_": cfg._target_, "cfg": cfg}, _recursive_=False)
    workspace.run()


if __name__ == "__main__":
    main()
