import torch
import random
import numpy 
import argparse
from dqn import QNetwork
from collections import namedtuple
from minatar import Environment
from tqdm import tqdm
import numpy as np
import time
import pickle
from dqn import replay_buffer, transition


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()

def world_dynamics(t, epsilon, num_actions, s, env, policy_net):

    if numpy.random.binomial(1, epsilon) == 1:  # with epsilon, output random action
        action = torch.tensor([[random.randrange(num_actions)]], device=device)
    else:
        with torch.no_grad():
            action = policy_net(s).max(1)[1].view(1, 1)
            
    reward, terminated = env.act(action)
    s_prime = get_state(env.state())

    return s_prime, action, torch.tensor([[reward]], device=device).float(), torch.tensor([[terminated]], device=device)


def main(args):
    env = Environment(args.game)

    in_channels = env.state_shape()[2]
    num_actions = env.num_actions()
    policy_net = QNetwork(in_channels, num_actions).to(device).eval()

    lt = time.time()
    checkpoint = torch.load(f"{args.root}/{args.game}/model/{args.load_path}")
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    print(f"Loading takes {time.time()-lt}s")

    savedict = {
        'obs': torch.zeros([args.replay_buffer_size, in_channels, 10, 10]),
        'action': torch.zeros([args.replay_buffer_size]),
        'reward': torch.zeros([args.replay_buffer_size]),
        'obs_prime': torch.zeros([args.replay_buffer_size, in_channels, 10, 10]),
        'done': torch.zeros([args.replay_buffer_size]),
        'mean_return': []
    }

    t = 0
    e = 0
    # data_return = []
    t_start = time.time()
    pbar = tqdm(total=args.replay_buffer_size)

    while t < args.replay_buffer_size:
        G = 0.0
        env.reset()
        s = get_state(env.state())
        is_terminated = False
        while(not is_terminated) and t < args.replay_buffer_size:
            # Generate data
            s_prime, action, reward, is_terminated = world_dynamics(t, args.eps, num_actions, s, env, policy_net)

            # breakpoint()
            savedict['obs'][t] = s[0]
            savedict['action'][t] = action[0]
            savedict['reward'][t] = reward[0]
            savedict['obs_prime'][t] = s_prime[0]
            savedict['done'][t] = is_terminated[0]

            G += reward.item()

            t += 1
            pbar.update(1)
            s = s_prime

        # Increment the episodes
        e += 1
        savedict['mean_return'].append(G)
        # data_return.append(G)

    print(f"{t} || {e} || {np.mean(savedict['mean_return'])} +- {np.std(savedict['mean_return'])/np.sqrt(len(savedict['mean_return']))} || {round(time.time()-t_start, 3)}s")

    if args.save:
        savepath = f"{args.root}/{args.game}/dataset/{args.game}_{args.eps}"
        print(savepath)
        with open(savepath, "wb") as f:
            pickle.dump(savedict, f)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default="freeway")  # breakout, freeway, seaquest, asterix
    parser.add_argument('--root', type=str, default="")
    parser.add_argument('--eps', type=float, default=0.5)
    parser.add_argument('--replay_buffer_size', '-rbs', type=int, default=1_000_000)
    parser.add_argument('--load_path', type=str, default='model_data_and_weights')
    parser.add_argument('--save', type=int, default=1)
    args = parser.parse_args()
    main(args)






