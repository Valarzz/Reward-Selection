import numpy as np

class il:
    def set_init(self, param_dict):
        self.adim = param_dict['total_actions']
        self.actions = np.arange(self.adim)
        self.i2s = param_dict['i2s']
        self.s2i = param_dict['s2i']
        
        dataset = param_dict['dataset']
        diff_keys = param_dict['diff_keys']
        all_keys = param_dict['frequency_percentage']
        self.sdim = len(all_keys) - len(diff_keys)

        self.frequency_percentage = np.zeros((self.sdim, self.adim))
        states = dataset['states']
        actions = dataset['actions']
        for s, a in zip(states, actions):
            s_idx = self.s2i[s]
            self.frequency_percentage[s_idx, a] += 1

        # Normalize to get probabilities
        row_sums = self.frequency_percentage.sum(axis=1, keepdims=True) + 1e-8
        self.frequency_percentage = self.frequency_percentage / row_sums

        self.all_state = np.arange(self.sdim)

    def train(self, train_state):

        train_state = np.array(train_state)
        diff = np.setdiff1d(self.all_state, train_state).astype(int)
        self.diff = diff

        return None if diff.size == 0 else il_agent(self.actions, self.frequency_percentage, diff, self.s2i)
    

class il_agent:
    def __init__(self, actions, frequency_percentage, seen_state, s2i):
        self.Q = None
        self.actions = actions
        self.frequency_percentage = frequency_percentage
        self.seen_state = seen_state
        self.s2i = s2i

    def policy(self, s):
        s = int(self.s2i[int(s)])
        return np.random.choice(self.actions, p=self.frequency_percentage[s]) 
        

    

