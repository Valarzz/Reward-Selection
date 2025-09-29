import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import copy
import random
from tqdm import tqdm, trange
from .utils import DEFAULT_DEVICE, expectile_loss, update_exponential_moving_average, IQLDataset


class QNetwork(nn.Module):
    def __init__(self, tau, beta, alpha, batch_size, num_epochs,):
        super().__init__()
        self.tau = tau
        self.beta = beta
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def set_init(self, param_dict, 
                 learning_rate=3e-4, ):

        in_channels = param_dict['env'].in_channels
        num_actions = param_dict['env'].num_actions_

        qf=TwinQ(in_channels, num_actions)
        vf=ValueFunction(in_channels)
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=learning_rate)
        num_actions=num_actions

        self.qf = qf.to(DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = vf.to(DEFAULT_DEVICE)

        self.v_optimizer = optimizer_factory(self.vf.parameters())
        self.q_optimizer = optimizer_factory(self.qf.parameters())
        self.num_actions = num_actions

        self.discount = param_dict['gamma']

    def forward(self, state, epsilon=0.05):
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        with torch.no_grad():
            q_values = self.qf(state.to(DEFAULT_DEVICE))
            action = q_values.argmax().view(1, 1)
        return action
    
    def trainn(self, dataset, train_indices):
        self.dataset = {k: torch.from_numpy(v).to(DEFAULT_DEVICE) for k, v in dataset.items()}
        self.dataset['actions'] = self.dataset['actions'].flatten()

        # train_indices_tensor = torch.from_numpy(train_indices, dtype=torch.bool)
        train_indices_tensor = train_indices
        print(len(train_indices_tensor))
        self.dataset = {k: v[train_indices_tensor] for k, v in self.dataset.items()}

        self.total_sample = len(self.dataset['states'])

        # for step in range(self.num_epochs):
        for step in trange(self.num_epochs):
            self.train()
            self.update(**self.sample_batch())

    def sample_batch(self):
        indices = torch.randint(low=0, high=self.total_sample, size=(self.batch_size,))
        return {
            'obs': self.dataset['states'][indices],
            'action': self.dataset['actions'][indices],
            'reward': self.dataset['rewards'][indices],
            'obs_prime': self.dataset['next_states'][indices],
            'done': self.dataset['dones'][indices]
        }
        # batch = {}
        # for key in ['obs', 'action', 'reward', 'obs_prime', 'done']:
        #     batch[key] = self.dataset[key][indices].to(DEVICE)
        # return batch

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
    







# class QNetwork(nn.Module):
#     def __init__(self, tau, beta, alpha, batch_size, num_epochs):
#         super().__init__()

#         self.tau = tau
#         self.beta = beta
#         self.alpha = alpha
#         self.batch_size = batch_size
#         self.num_epochs = num_epochs
#         self.num_samples = num_epochs * batch_size

#     def set_init(self, param_dict, ):
        
#         self.policy = iql_component()
#         self.policy.set_init(param_dict)

#         self.discount = param_dict['gamma']

#     def train(self, dataset):

#         self.temp_policy = copy.deepcopy(self.policy)
#         self.temp_policy.train()

#         # self.temp_policy.eval()
#         # return self.temp_policy

#         self.dataset_gpu = {k: torch.from_numpy(v).to(DEFAULT_DEVICE) for k, v in dataset.items()}
#         self.dataset_len = self.dataset_gpu['states'].shape[0]

#         for step in trange(self.num_epochs):
#         # for step in range(self.num_epochs):
#             self.update(**self.sample_batch_gpu())


#         # iqldataset = IQLDataset(dataset, self.num_samples)
#         # pin_memory = False if torch.cuda.is_available() else True
#         # data_loader = DataLoader(
#         #     iqldataset, 
#         #     batch_size=self.batch_size, 
#         #     shuffle=True,
#         #     pin_memory=pin_memory,
#         # )

#         # # for batch in data_loader:
#         # for batch in tqdm(data_loader):
#         #     self.update(batch)

#         self.temp_policy.eval()
#         return self.temp_policy

#     # def update(self, batch):

#     #     obs = batch['states']
#     #     action = batch['actions']
#     #     reward = batch['rewards']
#     #     obs_prime = batch['next_states']
#     #     done = batch['dones']

#     def update(self, obs, action, reward, obs_prime, done):
        
#         with torch.no_grad():
#             target_q_sa = self.temp_policy.q_target(obs, action)
#             next_v = self.temp_policy.vf(obs_prime).squeeze()

#         # Update value function
#         v_s = self.temp_policy.vf(obs)
#         value_loss = expectile_loss(v_s, target_q_sa, tau=self.tau)
#         self.temp_policy.v_optimizer.zero_grad(set_to_none=True)
#         value_loss.backward()
#         self.temp_policy.v_optimizer.step()

#         # Update Q function
#         targets = reward + (1. - done.float()) * self.discount * next_v.detach()
#         qs = self.temp_policy.qf.both(obs, action)
#         q_loss = sum(F.mse_loss(q.squeeze(), targets) for q in qs) / len(qs)
#         self.temp_policy.q_optimizer.zero_grad(set_to_none=True)
#         q_loss.backward()
#         self.temp_policy.q_optimizer.step()

#         # Update target Q network
#         update_exponential_moving_average(self.temp_policy.q_target, self.temp_policy.qf, self.alpha)

#     def sample_batch_gpu(self):
#         idx = torch.randint(0, self.dataset_len, (self.batch_size,))
#         # print(len(idx), self.dataset_len)
#         # breakpoint()
        # return {
        #     'obs': self.dataset_gpu['states'][idx],
        #     'action': self.dataset_gpu['actions'][idx],
        #     'reward': self.dataset_gpu['rewards'][idx],
        #     'obs_prime': self.dataset_gpu['next_states'][idx],
        #     'done': self.dataset_gpu['dones'][idx]
        # }
    
    
# class iql_component(nn.Module):
#     def __init__(self, ):
#         super().__init__()

#     def set_init(self, param_dict, 
#                  learning_rate=3e-4, ):
        
#         in_channels = param_dict['env'].in_channels
#         num_actions = param_dict['env'].num_actions_
        
#         qf = TwinQ(in_channels, num_actions)
#         self.qf = qf.to(DEFAULT_DEVICE)
#         self.q_target = copy.deepcopy(qf).requires_grad_(False).to(DEFAULT_DEVICE)

#         vf = ValueFunction(in_channels)
#         self.vf = vf.to(DEFAULT_DEVICE)

#         optimizer_factory=lambda params: torch.optim.Adam(params, lr=learning_rate)

#         self.v_optimizer = optimizer_factory(self.vf.parameters())
#         self.q_optimizer = optimizer_factory(self.qf.parameters())
#         self.num_actions = num_actions


