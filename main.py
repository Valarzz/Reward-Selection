import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import hydra
from hydra.utils import instantiate
from easydict import EasyDict
from omegaconf import OmegaConf
import yaml
import concurrent.futures
import time
import datetime
import uuid
from utils import *


def get_minatar_params(game):
    if game == 'breakout':  
        stepsize = 250
        game_budget = 12_001
    elif game == 'freeway':  
        stepsize = 15_000
        game_budget = 720_001
    elif game == 'seaquest':  
        stepsize = 12_000
        game_budget = 576_001
    elif game == 'asterix':  
        stepsize = 9_000
        game_budget = 432_001
    return stepsize, game_budget

def run_minatar_experiment(args, seed):
    game = args.domain.domain.game
    expname = args.general.expname
    algo = args.domain.exp.algo
    
    # Check if this is an evolutionary search case
    if args.selection == 'training_phase':
        if args.selection_params.search != 'evolutionary':
            raise ValueError("In minatar, when selection is 'training_phase', search must be 'evolutionary'")
        cmd = f"python minatar/esmain.py --root {args.root} --game {game} --budget {args.domain.exp.budget} --expname {expname} --M {args.domain.exp.M} --K {args.domain.exp.K} --E {args.domain.exp.E} --learning_rate {args.domain.exp.learning_rate} --alpha {args.domain.exp.alpha} --tau {args.domain.exp.tau} --beta {args.domain.exp.beta} --discount {args.domain.exp.discount} --train_steps {args.domain.exp.train_steps} --batch_size {args.domain.exp.batch_size} --eval_period {args.domain.exp.eval_period} --n_eval_episodes {args.domain.exp.n_eval_episodes} --each_query {args.domain.exp.each_query} --decay_temp {args.domain.exp.decay_temp} --initial_sample {args.domain.exp.initial_sample} --seed {seed}"
        os.system(cmd)

    else:
        stepsize, game_budget = get_minatar_params(game)
        
        if algo in ['uniform', 'visit']:
            uniform = 1 if algo == 'uniform' else 0
            cmd = f"python minatar/quickmain.py --root {args.root} --game {game} --budget {args.domain.exp.budget} --expname {expname} --seed {seed} --blind 1 --uniform {uniform}"
        else:
            cmd = f"python minatar/quickmain.py --root {args.root} --game {game} --budget {game_budget} --expname {expname} --seed {seed} --each_query {stepsize} --initial_sample {stepsize}"
        
        os.system(cmd)

def run_experiment(cfg, args, seed):
    # Check if this is a MinAtar domain before any domain imports
    if args.domain.domain.game in ['breakout', 'freeway', 'seaquest', 'asterix']:
        run_minatar_experiment(args, seed)
        return

    # Only import domains if not a MinAtar experiment
    import domains
    
    set_path(args)
    control_seed(seed)

    env = instantiate(args.domain.domain)
    qfunction = instantiate(args.domain.exp.qfunction)
    il = instantiate(args.domain.exp.il)
    experiment = instantiate(args.selection)
    
    dataset = get_dataset(env, args.dataset, args.root)
    
    experiment.init_exp(env, dataset, il, qfunction, args.domain.exp, 
                        args.root, args.selection_params, args.general, seed,
                        args.dataset)

    now_str = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    uid = str(uuid.uuid4())[:8]
    my_root_path = f"{experiment.save_root}/configs/{now_str}_{uid}"
    os.makedirs(my_root_path, exist_ok=True)
    config_path = os.path.join(my_root_path, "config.yaml")
    OmegaConf.save(cfg, config_path)

    experiment.run()



@hydra.main(config_path="config", config_name="config", version_base=None)
def main(hydra_cfg):
    
    yaml_config = OmegaConf.to_yaml(hydra_cfg, resolve=True)
    args = EasyDict(yaml.safe_load(yaml_config))

    if args.general.parallel_num == 1:
        run_experiment(hydra_cfg, args, args.general.seed)

    else:
        seeds = range(args.general.seed, args.general.seed + args.general.parallel_num)

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.general.parallel_num) as executor:
            futures = [
                executor.submit(run_experiment, hydra_cfg, args, seed)
                for seed in seeds
            ]
            # Wait for all tasks to complete (handle exceptions if needed)
            concurrent.futures.wait(futures)

    
if __name__ == "__main__" and __package__ is None:
    __package__ = "RLLF"
    main()
    

# 