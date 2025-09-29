import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import argparse
from minatar import Environment
from tqdm import trange
import numpy as np

from iql import cnn, sample_batch, evaluate_policy, plot_line

import time
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BehaviorCloningModel(nn.Module):
    def __init__(self, in_channels, num_actions, learning_rate=3e-4):
        super().__init__()
        self.bc = cnn(in_channels, num_actions)
        self.optimizer = optim.Adam(self.bc.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, state, eps=0):
        prob = torch.softmax(self.bc(state), dim=1)
        index = torch.multinomial(prob, num_samples=1)
        return index
    
    def update(self, obs, action, reward, obs_prime, done):
        self.bc.train()
        self.optimizer.zero_grad()
        outputs = self.bc(obs)
        # breakpoint()
        loss = self.criterion(outputs, action.long())
        loss.backward()
        self.optimizer.step()
        return loss.cpu().detach().item()


def main(args):
    env = Environment(args.game)
    in_channels = env.state_shape()[2]
    num_actions = env.num_actions()

    loadpath = f"{args.root}/{args.game}/dataset/{args.game}_"
    

    slt = time.time()
    if args.dataset == 'expert':
        with open(loadpath + "0.0", "rb") as f:
            dataset = pickle.load(f)
    elif args.dataset == 'rand':
        with open(loadpath + "1.0", "rb") as f:
            dataset = pickle.load(f)

    elif args.dataset == 'mix':
        with open(loadpath + "0.5", "rb") as f:
            dataset = pickle.load(f)

    print(f"Loading dataset take {time.time() - slt}s")  

    model = BehaviorCloningModel(in_channels, num_actions).to(device)

    # model.update(**sample_batch(dataset, args.batch_size))
    # J = evaluate_policy(env, model,  args.n_eval_episodes)
    # breakpoint()

    allJ = []
    for step in trange(args.n_steps):
        loss = model.update(**sample_batch(dataset, args.batch_size))

        if (step+1) % args.eval_period == 0:
            J = evaluate_policy(env, model, args.n_eval_episodes)
            allJ.append(J)
            print(loss, np.mean(J))

            plt.figure(figsize=(8, 6))
            plot_line(allJ, "All")
            plt.savefig(f"{args.game}plot/bc_train_{args.dataset}.pdf")
            plt.close()

    # breakpoint()
    Js = evaluate_policy(env, model, 50)
    # breakpoint()

    torch.save({'iql': model.state_dict(), 
                'Js': Js},
               f"{args.root}/{args.game}/result/BC_{args.dataset}.pts")
    


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--game', type=str, default="breakout")  # breakout, freeway, seaquest, asterix
    parser.add_argument('--root', type=str, default="")
    parser.add_argument('--dataset', '-dp', type=str, default="mix")  # expert, rand, mix
    
    parser.add_argument('--n_steps', '-n', type=int, default=10_000)  # 1_000_000, 1_000
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--eval_period', type=int, default=500)  # 5_000, 50
    parser.add_argument('--n_eval_episodes', type=int, default=5)

    args = parser.parse_args()
    main(args)


