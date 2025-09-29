import numpy as np
import random
import torch
from tqdm import tqdm
import h5py
import os
import json
import io


def control_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def set_path(args):
    root = args.root
    os.makedirs(root, exist_ok=True)

    def add_root_to_path(args, name):
        if hasattr(args, name):
            setattr(args, name, f"{root}/{getattr(args, name)}")

    add_root_to_path(args.domain.domain, "expert_policy_path")
    add_root_to_path(args.domain.exp.il, "load_path")



def get_dataset(env, args, root):

    if args.dc_ratio is not None: 
        # In-memory dataset
        dataset =  make_dataset(env, None, args)
        return dataset

    else: 
        dataset_path = f"{root}/{env.name}/dataset/{args.data_collecting}_{args.dataset_seed}.h5"
        if os.path.exists(dataset_path):
            print(f"found dataset at {dataset_path}")
        else:
            make_dataset(env, dataset_path, args)
        return h5py.File(dataset_path, 'r')

def make_dataset(env, dataset_path, args):
    print(f"no dataset found at {dataset_path}, creating one ...")
    if dataset_path is not None:
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    control_seed(args.dataset_seed)
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    state_vistation = {}
    next_state_vistation = {}
    episode_returns = []

    with tqdm(total=args.dataset_size) as pbar:
        while len(states) < args.dataset_size:
            state, _ = env.reset()
            done = False
            episode_return = 0
            steps = 0

            # while not done:
            while (not done) and (steps < env.MAX_STEPS):
                steps += 1
                if args.dc_ratio is None:
                    action = env.create_dataset(args.data_collecting, state) 
                else:
                    action = env.create_dataset(args.data_collecting, state, float(args.dc_ratio))
                next_state, reward, done, _ = env.step(action)
                episode_return += reward

                states.append(state)
                next_states.append(next_state)
                ind = len(states) - 1

                state_vistation.setdefault(state, []).append(ind)
                next_state_vistation.setdefault(next_state, []).append(ind)
                
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                
                pbar.update(1)
                state = next_state

            episode_returns.append(episode_return)

    rewards = np.array(rewards)
    episode_returns = np.array(episode_returns)
    print(rewards.min(), rewards.max(), rewards.mean(), episode_returns.mean())
    
    json_ready_state_vistation = {
        str(tuple(int(x) for x in k)) if isinstance(k, tuple) else str(k): v 
        for k, v in state_vistation.items()
    }
    json_ready_next_state_vistation = {
        str(tuple(int(x) for x in k)) if isinstance(k, tuple) else str(k): v 
        for k, v in next_state_vistation.items()
    }

    if dataset_path is not None:
        with h5py.File(dataset_path, 'w') as h5file:
            h5file.create_dataset('states', data=np.array(states))
            h5file.create_dataset('actions', data=np.array(actions))
            h5file.create_dataset('rewards', data=rewards)
            h5file.create_dataset('next_states', data=np.array(next_states))
            h5file.create_dataset('dones', data=np.array(dones))
            h5file.create_dataset('state_vistation', data=json.dumps(json_ready_state_vistation))
            h5file.create_dataset('next_state_vistation', data=json.dumps(json_ready_next_state_vistation))
        print(f"dataset created at {dataset_path}")
    else:
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': rewards,
            'next_states': np.array(next_states),
            'dones': np.array(dones),
            'state_vistation': state_vistation,
            'next_state_vistation': next_state_vistation,
        }

