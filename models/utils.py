import copy

import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import time


DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_batched(f, xs):
    return f(torch.cat(xs, dim=0)).split([len(x) for x in xs])


def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)


################################# UTILS #####################################
        
def expectile_loss(pred, target, tau=0.7):
    """
    pred:  (batch,)
    target: (batch,)
    tau: scalar in (0,1)
    """
    diff = target - pred
    # Weighted MSE depending on sign of diff
    weight = torch.where(diff > 0, torch.tensor(tau), torch.tensor(1.0 - tau)).to(diff.device)
    loss = weight * (diff ** 2)
    return loss.mean()

# dataset is a dict, values of which are tensors of same first dimension
def sample_batch(dataset, batch_size):
    # samplest = time.time()
    N = len(dataset['obs'])
    indices = torch.randint(low=0, high=N, size=(batch_size,))

    batch = {}
    for key in ['obs', 'action', 'reward', 'obs_prime', 'done']:
        batch[key] = dataset[key][indices].to(DEFAULT_DEVICE)

    # print(time.time() - samplest)
    return batch


def get_state(s):
    return (torch.tensor(s, device=DEFAULT_DEVICE).permute(2, 0, 1)).unsqueeze(0).float()

def world_dynamics(s, env, policy_net, eps):
    action = policy_net(s, eps)
            
    reward, terminated = env.act(action)
    s_prime = get_state(env.state())

    return s_prime, action, torch.tensor([[reward]], device=DEFAULT_DEVICE).float(), torch.tensor([[terminated]], device=DEFAULT_DEVICE)

def torchify(x):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device=DEFAULT_DEVICE)
    return x

def evaluate_policy(env, policy, max_episode_steps, deterministic=True):
    data_return = []
    eps = 0. if deterministic else 0.05
    for _ in tqdm(range(max_episode_steps)):
    # for _ in range(max_episode_steps):
        with torch.no_grad():
            G = 0.0
            env.reset()
            s = get_state(env.state())
            is_terminated = False
            while(not is_terminated):
                # Generate data
                s_prime, action, reward, is_terminated = world_dynamics(s, env, policy, eps)
                G += reward.item()
                s = s_prime
            data_return.append(G)
    return data_return

def plot_line(data, title):
    means = np.array([np.mean(arr) for arr in data])
    std_devs = np.array([np.std(arr)/np.sqrt(len(arr)) for arr in data])

    print(title, means.mean())
    df = pd.DataFrame({
        'Index': np.arange(0, len(means)),  # Ensure Index is numeric
        'Mean': means,
        'StdDev': std_devs
    })
    sns.lineplot(data=df, 
                 x='Index', y='Mean', 
                 label=title, 
                 )
    plt.fill_between(
        df['Index'],
        df['Mean'] - df['StdDev'],
        df['Mean'] + df['StdDev'],
        alpha=0.2,
    )

################################# Q & V #####################################

class cnn(nn.Module):
    def __init__(self, in_channels, out_dim):  # q:num_actions, v: 1
        super(cnn, self).__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)
        self.output = nn.Linear(in_features=128, out_features=out_dim)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = F.relu(self.conv(x))
        x = F.relu(self.fc_hidden(x.view(x.size(0), -1)))
        return self.output(x)


class TwinQ(nn.Module):
    def __init__(self, in_channels, num_actions):
        super().__init__()
        self.q1 = cnn(in_channels, num_actions)
        self.q2 = cnn(in_channels, num_actions)

    def both(self, state, action=None):
        if action is None:
            return self.q1(state), self.q2(state)
        else:
            q1_sa = self.q1(state).gather(1, action.long().squeeze(1))
            q2_sa = self.q2(state).gather(1, action.long().squeeze(1))
            return q1_sa, q2_sa

    def forward(self, state, action=None):
        if action is None:
            q1_sa, q2_sa = self.both(state, action)
            q12 = torch.cat((q1_sa, q2_sa), dim=0)
            minq = torch.min(q12, dim=0).values
            return minq
        else:
            return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.v = cnn(in_channels, 1)

    def forward(self, state):
        return self.v(state)


class qnet(nn.Module):
    def __init__(self, ):
        super(qnet, self).__init__()

    def set_init(self, param_dict):

        in_channels = param_dict['env'].in_channels
        num_actions = param_dict['env'].num_actions_

        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)

        # Output layer:
        self.output = nn.Linear(in_features=128, out_features=num_actions)

    # As per implementation instructions according to pytorch, the forward function should be overwritten by all
    # subclasses
    def forward(self, x):
        x = torch.from_numpy(x)
        if x.ndim == 3:
            x = x.unsqueeze(0)

        # Rectified output from the first conv layer
        x = F.relu(self.conv(x))

        # Rectified output from the final hidden layer
        x = F.relu(self.fc_hidden(x.view(x.size(0), -1)))

        # Returns the output from the fully-connected linear layer
        return self.output(x)


class IQLDataset(Dataset):
    def __init__(self, dataset, N):
        self.dataset = {k: torch.from_numpy(v).to(DEFAULT_DEVICE) for k, v in dataset.items()}
        self.N = N
        self.actual_length = len(dataset['states'])

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        real_idx = idx % self.actual_length
        return {
            'states': self.dataset['states'][real_idx],
            'actions': self.dataset['actions'][real_idx],
            'rewards': self.dataset['rewards'][real_idx],
            'next_states': self.dataset['next_states'][real_idx],
            'dones': self.dataset['dones'][real_idx],
        }


