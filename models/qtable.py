import numpy as np
from collections import defaultdict
from tqdm import trange
import random

class qtable:
    def set_init(self, param_dict):
        self.sdim = param_dict['total_states']
        self.adim = param_dict['total_actions']
        self.gamma = param_dict['gamma']
        self.qiterations = param_dict['qiterations']
        self.qalpha = param_dict['qalpha']
        self.qinit = param_dict['qinit']
        self.rmax_ucb = param_dict['rmax'] / (1 - param_dict['gamma'])
        self.rmin_ucb = param_dict['rmin'] / (1 - param_dict['gamma'])
        self.impute_type = param_dict['impute_type']

    def train(self, dataset, Q=None, ilagent=None, tol=1e-3):  # increased default tolerance
        train_inds = list(range(len(dataset['states'])))
        alpha = self.qalpha # 0.05
        num_epochs = self.qiterations #500
        
        if Q is None:
            if self.qinit == 'zero':
                Q = defaultdict(lambda: np.zeros(self.adim))
            elif self.qinit == 'random':
                Q = defaultdict(lambda: np.random.randn(self.adim))
            elif self.qinit == 'ucb_max':
                Q = defaultdict(lambda: np.ones(self.adim) * self.rmax_ucb)
            elif self.qinit == 'ucb_min':
                Q = defaultdict(lambda: np.ones(self.adim) * self.rmin_ucb)
        else:
            Q = Q

        prev_avg_loss = float('inf')  # Track previous epoch's loss

        for epoch in range(num_epochs):
        # for epoch in trange(num_epochs):
            total_loss = 0.0

            random.shuffle(train_inds)
            for ind in train_inds:

                s = int(dataset['states'][ind])
                a = int(dataset['actions'][ind])
                r = dataset['rewards'][ind]
                s_ = int(dataset['next_states'][ind])
                done = bool(dataset['dones'][ind])

                if self.impute_type == "none":
                    if (done) or (s_ not in Q.keys()):
                        V_s_next = 0.0
                    else:
                        V_s_next = Q[s_].max() 

                else:
                    if done:
                        V_s_next = 0.0
                    else:
                        V_s_next = Q[s_].max() 
                        # if ilagent is not None:  # q evaluation
                        #     V_s_next = (ilagent.frequency_percentage[s] * Q[s_]).sum()
                        # else:  # q learning
                        #     V_s_next = Q[s_].max() 

                Q_target = r + self.gamma * V_s_next

                q_error = Q_target - Q[s][a]
                Q[s][a] += alpha * q_error
                total_loss += abs(q_error)

            avg_loss = total_loss / (len(train_inds)+1)
            delta_loss = abs(prev_avg_loss - avg_loss)
            # Check for convergence
            if delta_loss < tol:
                # print(f"Converged at epoch {epoch+1} with loss difference: {delta_loss:.5f}")
                break
            
            prev_avg_loss = avg_loss

        # print(f" at epoch {epoch+1} with loss difference: {delta_loss:.5f}")
        # breakpoint()
        
        return dict(Q)
    