import os
import pickle
import numpy as np
import torch
from argparse import ArgumentParser
from minatar import Environment
from iql import ImplicitQLearning, TwinQ, ValueFunction
from bc import BehaviorCloningModel
from minatar.heuristics import heuristicselection, DEVICE

def control_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def save_file_es(args, budget, Js, dist_idx, iter_idx):
    savepath = f"{args.root}/{args.game}mdp/result/{args.expname}/es/{budget}/"
    os.makedirs(savepath, exist_ok=True)
    np.save(savepath+f"dist{dist_idx}_iter{iter_idx}.npy", np.array(Js))

def main(args):
    control_seed(args.seed)
    with open(f"{args.root}/{args.game}mdp/dataset/{args.game}_0.5", "rb") as f:
        dataset = pickle.load(f)
    env = Environment(args.game)
    in_channels = env.state_shape()[2]
    num_actions = env.num_actions()
    iql = ImplicitQLearning(
        qf=TwinQ(in_channels, num_actions),
        vf=ValueFunction(in_channels),
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
        num_actions=num_actions,
        tau=args.tau,
        beta=args.beta,
        alpha=args.alpha,
        discount=args.discount,
    ).to(DEVICE)
    bc = BehaviorCloningModel(in_channels, num_actions).to(DEVICE)
    selectionagent = heuristicselection(iql=iql,
                 bc=bc,
                 dataset=dataset, 
                 env=env, 
                 budget=args.budget, 
                 each_query=args.each_query,
                 sdim=in_channels,
                 adim=num_actions,
                 N=1000, 
                 gamma=args.discount, 
                 blind=False,  # ES is not blind
                 uniform=False, 
                 decay='linear',
                 zero='vis',
                 fixeddecay=False,
                 fixtime=0.5,
                 train_steps=args.train_steps,  # 200_000
                 eval_period=args.eval_period,
                 batch_size=args.batch_size,
                 n_eval_episodes=args.n_eval_episodes,
                 decay_temp=args.decay_temp,
                 initial_sample=args.initial_sample,
                 game=args.game,
                 )
    # Load indices
    indices = np.load(args.es_indices)
    indices = indices.tolist() if isinstance(indices, np.ndarray) else list(indices)
    Js, _ = selectionagent.iqltrain(indices, test=True)
    print(Js, np.mean(Js))
    save_file_es(args, args.budget, Js, args.dist_idx, args.iter_idx)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--expname', type=str, default="esdebug")
    parser.add_argument('--root', type=str, default="")
    parser.add_argument('--game', type=str, default="breakout")
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--train_steps', type=int, default=125_000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--eval_period', type=int, default=500)
    parser.add_argument('--n_eval_episodes', type=int, default=50)
    parser.add_argument('--budget', type=int, default=10000)
    parser.add_argument('--each_query', type=int, default=250)
    parser.add_argument('--decay_temp', type=float, default=6.)
    parser.add_argument('--initial_sample', type=int, default=250)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--es_indices', type=str, required=True)
    parser.add_argument('--dist_idx', type=int, required=True)
    parser.add_argument('--iter_idx', type=int, required=True)
    args = parser.parse_args()
    main(args) 