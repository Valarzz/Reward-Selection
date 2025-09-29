import copy

import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions import MultivariateNormal
import pandas as pd
import seaborn as sns
from tqdm import tqdm, trange
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_batched(f, xs):
    return f(torch.cat(xs, dim=0)).split([len(x) for x in xs])


def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)


class ImplicitQLearning(nn.Module):
    def __init__(self, qf, vf, optimizer_factory, num_actions,
                 tau, beta, discount=0.99, alpha=0.005):
        super().__init__()
        
        self.qf = qf.to(DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = vf.to(DEFAULT_DEVICE)

        self.v_optimizer = optimizer_factory(self.vf.parameters())
        self.q_optimizer = optimizer_factory(self.qf.parameters())
        self.num_actions = num_actions

        self.tau = tau
        self.beta = beta
        self.discount = discount
        self.alpha = alpha

    def forward(self, state, epsilon=0.05):
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        with torch.no_grad():
            q_values = self.qf(state.to(DEFAULT_DEVICE))
            action = q_values.argmax().view(1, 1)
        return action

    def update(self, obs, action, reward, obs_prime, done):
        with torch.no_grad():
            target_q_sa = self.q_target(obs, action)
            next_v = self.vf(obs_prime).squeeze()

        # Update value function
        v_s = self.vf(obs)
        value_loss = expectile_loss(v_s, target_q_sa, tau=self.tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        self.v_optimizer.step()

        # Update Q function
        targets = reward + (1. - done.float()) * self.discount * next_v.detach()
        qs = self.qf.both(obs, action)
        q_loss = sum(F.mse_loss(q.squeeze(), targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        update_exponential_moving_average(self.q_target, self.qf, self.alpha)




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
    steps = 0

    for _ in trange(max_episode_steps):
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
                steps += 1
                s = s_prime
            data_return.append(G)
            print(G, steps)
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
            q1_sa = self.q1(state).gather(1, action.long().unsqueeze(1))
            q2_sa = self.q2(state).gather(1, action.long().unsqueeze(1))
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



