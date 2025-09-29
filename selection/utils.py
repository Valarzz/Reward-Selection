import numpy as np
import random
import torch

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def control_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def runEpisode(env, agent):
    G = 0
    t = 0
    done = False
    steps = 0

    state, info = env.reset()
    while (not done) and (steps < env.MAX_STEPS):
        steps += 1
        action = agent.policy(state)
        state, reward, done, _ = env.step(action)
        # print(state, action, reward, done)
        G += reward
    # breakpoint()
    return G

class offlineRL:
    def __init__(self, Q, ilagent, dataset_states, anum, packbits, impute, s2i, softmax=False):
        self.Q = Q
        self.softmax = softmax
        self.ilagent = ilagent
        self.dataset_states = dataset_states.astype(int)
        self.anum = anum
        self.packbits = packbits
        self.impute = impute
        self.s2i = s2i
        self.debugflag = False
    
    def debug(self, ):
        self.debugflag = True
    
    def policy(self, s):

        if s not in self.dataset_states:
            return np.random.randint(0, self.anum)
        
        s_key_flag = (self.ilagent is not None)
        if s_key_flag:
            s_key_flag = s_key_flag and (self.s2i[s] in self.ilagent.seen_state)

        if self.impute is None:
            # breakpoint()
            if s_key_flag: 
                return self.dpolicy(s)
            else:
                return self.qpolicy(s)
            
        else:
            return self.qpolicy(s)

    def dpolicy(self, s):
        return self.ilagent.policy(s)

    def qpolicy(self, s):
        q_values = self.Q[s]
        max_value = np.max(q_values)

        if self.softmax:
            exp_q = np.exp(q_values-max_value)
            probs = exp_q / np.sum(exp_q)
            action = np.random.choice(len(q_values), p=probs)
        
        else:
            max_actions = np.where(q_values == max_value)[0]
            action = np.random.choice(max_actions)
            
        return action  

