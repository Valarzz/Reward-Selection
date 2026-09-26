import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import hydra
from hydra.utils import instantiate
from easydict import EasyDict
from omegaconf import OmegaConf
import yaml
import domains
import concurrent.futures
import time
from utils import *
import os


def make_minatar_dataset(args):
    cmd = f"python minatar/makedataset.py --game {args.domain.domain.game} --root {args.root} "      
    os.system(cmd)

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(hydra_cfg):
    
    yaml_config = OmegaConf.to_yaml(hydra_cfg, resolve=True)
    args = EasyDict(yaml.safe_load(yaml_config))

    if args.domain.domain.get('game') in ['breakout', 'freeway', 'seaquest', 'asterix']:
        make_minatar_dataset(args)
        return

    set_path(args)

    env = instantiate(args.domain.domain)
    
    get_dataset(env, args.dataset, args.root)

    
if __name__ == "__main__" and __package__ is None:
    __package__ = "RLLF"
    main()
    
