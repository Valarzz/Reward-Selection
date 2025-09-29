from minatar.heuristics import heuristicselection, DEVICE
import pickle
from minatar import Environment
from iql import ImplicitQLearning, TwinQ, ValueFunction
import torch
from bc import BehaviorCloningModel
import numpy as np
from filelock import FileLock
import os
import time
from ipdb import launch_ipdb_on_exception
from tqdm import trange
import random
import h5py


def control_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def addkey(d, k, budget, Js, policy):
    if k not in d.keys():  # add decay policy
        d[k] = {}

    if budget not in d[k].keys():  # each decay policy, add budget
        d[k][budget] = {
            'Js': [Js],
            # 'policy': [policy],
        }
    else:
        d[k][budget]['Js'].append(Js)
        # d[k][budget]['policy'].append(policy)


def main(args):
    # print("Preparing env ... ")
    env = Environment(args.game)
    in_channels = env.state_shape()[2]
    num_actions = env.num_actions()

    print(in_channels, num_actions)
    # print("Done") 

    control_seed(args.seed)

    st = time.time()
    print("Loading dataset ...")
    loadpath = f"{args.root}/{args.game}/dataset/{args.game}_0.5"
    with open(loadpath, "rb") as f:
        dataset = pickle.load(f)
    print(f"Done: takes {time.time() - st}s")

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

                 blind=args.blind,  # test with blind first
                 uniform=args.uniform, 
                 decay=args.decay,
                 zero=args.zero,
                 fixeddecay=args.fixeddecay,
                 fixtime=args.fixtime,

                 train_steps=args.train_steps,  # 200_000
                 eval_period=args.eval_period,
                 batch_size=args.batch_size,
                 n_eval_episodes=args.n_eval_episodes,

                 decay_temp=args.decay_temp,
                 initial_sample=args.initial_sample,

                 game=args.game,
                 )

    print("Start training!!")
    if selectionagent.blind: # or (selectionagent.initial_sample>selectionagent.budget):
        next_visit_ids = selectionagent.get_state(args.budget)
        Js, policy = selectionagent.iqltrain(next_visit_ids, test=True)
        save_file(args, args.budget, Js, policy)
        print(np.mean(Js))

    else:
        # for budget in trange(selectionagent.initial_sample, selectionagent.budget+selectionagent.each_query, selectionagent.each_query):
        for budget in range(selectionagent.initial_sample, args.budget+selectionagent.each_query, selectionagent.each_query):

            budget = min(budget, args.budget)
            
            next_visit_ids = selectionagent.get_state(budget)
            Js, policy = selectionagent.iqltrain(next_visit_ids, test=True)

            save_file(args, budget, Js, policy)
            print(np.mean(Js), budget)
            print(Js)

    
    print(f"Total takes {time.time()-st}s")

    # exit()
def save_file(args, budget, Js, policy):

    if args.blind:
        if args.uniform:
            key = "uniform"
        else:
            key = "non-uniform"

    else:
        if args.decay == 'weird_linear':
            key = str(args.alpha)
        else:
            if args.fixeddecay:
                key = args.decay + f'-fixed-{args.fixtime}-temp-{args.decay_temp}'
            else:
                key = args.decay
        
        if args.zero == 'vis':
            key = 'nonu-' + key
        else:
            key = 'u-' + key

        key = key + f'-is{args.initial_sample}'

    if not args.uil:
        key = 'noIL-' + key

    savepath = f"{args.root}/{args.game}mdp/result/{args.expname}/{key}/{budget}/"
    os.makedirs(savepath, exist_ok=True)

    np.save(savepath+f"{args.seed}.npy", np.array(Js))
    


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--expname', type=str, default="debug")

    parser.add_argument('--root', type=str, default="")

    parser.add_argument('--game', type=str, default="breakout")  # breakout, freeway, seaquest, asterix
    parser.add_argument('--dataset', type=str, default="mix")  # expert, rand, mix

    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--discount', type=float, default=0.99)

    parser.add_argument('--train_steps', type=int, default=125_000)  # 1_000_000, 1_000
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--eval_period', type=int, default=500)  # No use
    parser.add_argument('--n_eval_episodes', type=int, default=50)

    parser.add_argument('--budget', type=int, default=10000)  # [2000, 4000, 6000, 8000, 10000]
    parser.add_argument('--each_query', type=int, default=250)  # 250
    parser.add_argument('--blind', type=int, default=0)
    parser.add_argument('--uniform', type=int, default=0)
    parser.add_argument('--decay', type=str, default='concave')  # convex, concave, linear
    parser.add_argument('--fixeddecay', type=int, default=0) 
    parser.add_argument('--fixtime', type=float, default=0.5) 
    parser.add_argument('--zero', type=str, default='vis')
    parser.add_argument('--uil', type=int, default=1)

    parser.add_argument('--decay_temp', type=float, default=6.)
    parser.add_argument('--initial_sample', type=int, default=250)
    parser.add_argument('--seed', type=int, default=41)

    args = parser.parse_args()
    print(args)
    print(DEVICE)

    main(args) 

    # with launch_ipdb_on_exception():
    #     main(args)

# python quickmain.py --blind 1 --uniform 1 --budget 2000 
# python quickmain.py --blind 1 --uniform 0 --game asterix --budget 13125 
# python quickmain.py --blind 1 --uniform 0 --budget 10000 --decay concave --zero vis  --expname try2
# python main.py --blind 0 --uniform 0 --budget 2000 --decay concave --zero vis  --expname try2


# python quickmain.py --blind 1 --uniform 0 --game breakout --budget 13731 
# python quickmain.py --blind 1 --uniform 0 --game freeway --budget 749824 
# python quickmain.py --blind 1 --uniform 0 --game seaquest --budget 601543 
# python quickmain.py --blind 1 --uniform 0 --game asterix --budget 480185 

# python quickmain.py --blind 0 --game breakout --each_query 250 --initial_sample 250
# python quickmain.py --blind 0 --game freeway --each_query 15000 --initial_sample 15000
# python quickmain.py --blind 0 --game seaquest --each_query 12000 --initial_sample 12000
# python quickmain.py --blind 0 --game asterix --each_query 10000 --initial_sample 10000

# breakout:  13_731        250
# freeway:  749_824     15_000
# seaquest: 601_543     12_000
# asterix:  480_185     10_000
