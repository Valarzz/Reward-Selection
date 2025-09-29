import numpy as np
import json
import ast
import os
import copy
from copy import deepcopy
from .utils import runEpisode, offlineRL, control_seed, DEFAULT_DEVICE
from tqdm import trange


class base_selection:
    def __init__(self, ):
        pass

    def init_exp(self, env, dataset, il, qfunction, args, root, 
                 selection_params, general_params, seed, dataset_params):
        
        self.env = env

        self.each_query = args.each_query
        self.warmup_sample = args.warmup_sample
        self.decay = args.decay
        self.fixeddecay = args.fixeddecay
        self.fixtime = args.fixtime
        self.decay_temp = args.decay_temp
        self.packbits = args.packbits
        self.algo = args.algo
        self.impute_type = args.impute

        self.softmax_af = args.softmax_af
        self.af_thre = args.af_thre

        self.eval_episodes = selection_params.eval_episodes
        self.gamma = selection_params.gamma
        self.search = selection_params.search
        self.beam_size = selection_params.beam_size
        self.visit_ids = selection_params.visit_ids
        self.evo_num = selection_params.evo_num
        self.evo_iter = selection_params.evo_iter
        self.tt = selection_params.tt
        self.evo_estimate = selection_params.evo_estimate
        
        self.dataset_seed = dataset_params.dataset_seed
        self.dc_ratio = dataset_params.dc_ratio
        self.reward_type = env.reward_type

        self.traintest = selection_params.tt
        self.train_ratios = selection_params.train
        self.test_ratios = selection_params.test 

        self.seed = seed
        self.root = root

        self.expname = general_params.expname
        self.budget = general_params.budget
        self.save_result = general_params.save_result
        self.qinit = general_params.qinit
        # breakpoint()
        rewards = np.array(dataset['rewards'])
        self.rmax = rewards.max()
        self.rmin = rewards.min()
        # print(rewards.min(), rewards.max(), rewards.mean())
        # breakpoint()
        if self.impute_type == "none":
            self.impute = None
        elif self.impute_type == "zero":
            self.impute = 0

        elif self.impute_type == "mean":
            self.impute = rewards.mean()
        elif self.impute_type == "max":
            self.impute = self.rmax
        elif self.impute_type == "min":
            self.impute = self.rmin
        else:
            raise NotImplementedError(f"Bad impute {self.impute_type}")
        
        # this is only for training, original data format
        self.dataset = {
            'states': np.array(dataset['states']),
            'actions': np.array(dataset['actions']),
            'rewards': rewards,
            'next_states': np.array(dataset['next_states']),
            'dones': np.array(dataset['dones'])
        }  

        self.obs_bits = self.dataset['states']
        self.obs_prime_bits = self.dataset['next_states']

        if self.dc_ratio is None:
            self.state_visitation = self.key2int(json.loads(dataset
            ['state_vistation'][()]))
            self.next_state_visitation = self.key2int(json.loads(dataset
            ['next_state_vistation'][()]))
        else:
            self.state_visitation = dataset['state_vistation']
            self.next_state_visitation = dataset['next_state_vistation']
        
        self.unique_obs_keys_notdone = list(self.state_visitation.keys())

        self.diff_keys = [k for k in self.next_state_visitation.keys() if k not in self.unique_obs_keys_notdone]
        self.unique_obs_keys = self.unique_obs_keys_notdone + self.diff_keys
        
        self.unique_obs = np.array(self.unique_obs_keys)  
        self.i2s = {i:s for i, s in enumerate(self.unique_obs)}
        self.s2i = {s:i for i, s in enumerate(self.unique_obs)}

        self.total_states = len(self.unique_obs)
        self.total_actions = env.num_actions()
        
        unique_obs_counts = np.zeros(self.total_states)  # id to distribution
        for s, i in self.s2i.items():
            if s in list(self.state_visitation.keys()):
                unique_obs_counts[i] = len(self.state_visitation[s])
        self.freq = unique_obs_counts / (unique_obs_counts.sum() + 1e-6)

        self.total_sample = len(self.obs_bits)
        self.visited_ids = []
        self.train_inds = np.full((self.total_sample), False, dtype=bool)
        self.train_inds_list = []

        param_dict = {
            'env': env,
            'total_actions': self.total_actions,
            'gamma': self.gamma,
            'total_states': self.total_states,
            'dataset': self.dataset,

            'qiterations': args.qiterations,
            'qalpha': args.qalpha,

            'rmax': self.rmax,
            'rmin': self.rmin,
            'qinit': self.qinit,
            'frequency_percentage': self.freq,
            'impute_type': self.impute_type,
            'diff_keys': self.diff_keys,
            'i2s': self.i2s,
            's2i': self.s2i,
        }

        self.qfunction = qfunction
        self.qfunction.set_init(param_dict)
        if self.impute_type == "none":
            self.il = il
            self.il.set_init(param_dict)
        else:
            self.il = None

        if self.impute is not None:
            self.bestq = self.qfunction.train(self.dataset)

        if self.warmup_sample:
            self.initial_sample = int(self.warmup_sample * (self.total_states - len(self.diff_keys)) + 1)
            
            prob = np.ones(self.total_states) 
            prob[-len(self.diff_keys):] = 0
            prob = prob / prob.sum()
            sampled_inds = np.random.choice(self.total_states, p=prob, size=self.initial_sample, replace=False)
            next_visit_ids = sampled_inds.tolist()

            iql_dataset_indices, zero_indice = self.get_sample_inds(next_visit_ids)
            self.train_inds = np.logical_or(self.train_inds, iql_dataset_indices)
            self.train_inds_list += zero_indice
            self.visited_ids += next_visit_ids

        else:
            self.initial_sample = 0

        self.current_Q = {s:np.zeros(self.total_actions) for s in self.unique_obs}

        # estimate dynamic function
        counts = np.zeros((self.total_states, self.total_actions, self.total_states), dtype=np.int64)
        for s, a, sp in zip(self.dataset["states"],
                            self.dataset["actions"],
                            self.dataset["next_states"]):
            counts[self.s2i[s], a, self.s2i[sp]] += 1
        P_hat = counts 
        P_hat = P_hat / (P_hat.sum(axis=-1, keepdims=True) + 1e-12)
        self.P_hat = P_hat

        # estimate d0
        starts = np.zeros(self.total_states, dtype=np.int64)
        dones  = self.dataset["dones"]
        episode_start_idxs = [0] + list(np.where(dones[:-1])[0] + 1)
        for idx in episode_start_idxs:
            start_state = dataset["states"][idx]
            start_state_id = self.s2i[start_state]
            starts[start_state_id] += 1
        d0 = starts / (starts.sum() + 1e-6)
        self.d0 = d0

        self.save_root = f"{root}/{self.env.name}/result/{self.selection_name}/{self.expname}"
        os.makedirs(self.save_root, exist_ok=True)
    
    def key2int(self, inp_dict):
        oup_dict = {}
        for k, v in inp_dict.items():
            oup_dict[int(k)] = v
        return oup_dict
    
    def get_sample_inds(self, next_visit_ids):
        selected_state_bits = self.unique_obs[next_visit_ids]
        # breakpoint()

        iql_dataset_indices = np.zeros(self.total_sample, dtype=bool)
        zero_indice = []
        for selected_state in selected_state_bits:
            selected_state = selected_state
            zero_indice += self.state_visitation[selected_state]
        iql_dataset_indices[zero_indice] = True

        return iql_dataset_indices, zero_indice

    def get_imputed_dataset(self, train_inds_list, impute_value):
        all_indices = set(range(self.total_sample))
        train_indices_set = set(train_inds_list)
        indices_to_impute = list(all_indices - train_indices_set)

        impute_iqldataset = deepcopy(self.dataset)
        impute_iqldataset['rewards'][indices_to_impute] = impute_value
        return impute_iqldataset

    def discounted_occupancy(self, Q, tol=1e-6, max_iter=10_000):
        def max_indicator(arr):
            arr = np.asarray(arr)
            max_val = np.max(arr)
            return (arr == max_val).astype(int) / self.total_actions

        P_pi = np.zeros((self.total_states, self.total_states), dtype=np.float64)
        for i, s in self.i2s.items():
            if s in Q.keys():
                probs = max_indicator(Q[s])          # shape [A]
                #   P_hat[i_s] : shape [A, S]
                P_pi[i] = (probs[:, None] * self.P_hat[i]).sum(axis=0)
            else:
                P_pi[i][i] = 1

        G = deepcopy(self.d0)
        for _ in range(max_iter):
            G_new = self.d0 + self.gamma * P_pi.T @ G
            if np.max(np.abs(G_new - G)) < tol:
                break
            G = G_new
        return G

    def discounted_occupancy_time(self, Q, horizon=1000, tol=1e-6, max_iter=10_000):
        """Compute time-dependent discounted occupancy measure d_π(s,t) and sum over t > 0.5T.
        
        Args:
            Q: State-action value function
            horizon: Number of timesteps to track
            tol: Convergence tolerance
            max_iter: Maximum iterations for convergence
            
        Returns:
            Array of shape [S] containing summed occupancy measures for t > horizon/2
        """
        def max_indicator(arr):
            arr = np.asarray(arr)
            max_val = np.max(arr)
            return (arr == max_val).astype(int) / self.total_actions

        # Get policy transition matrix
        P_pi = np.zeros((self.total_states, self.total_states), dtype=np.float64)
        for i, s in self.i2s.items():
            if s in Q.keys():
                probs = max_indicator(Q[s])          # shape [A]
                P_pi[i] = (probs[:, None] * self.P_hat[i]).sum(axis=0)
            else:
                P_pi[i][i] = 1

        # Initialize time-dependent occupancy matrix [T, S]
        G = np.zeros((horizon, self.total_states))
        G[0] = deepcopy(self.d0)  # Initial state distribution

        # Compute time-dependent occupancy measures with convergence check
        for t in range(1, horizon):
            G[t] = P_pi.T @ G[t-1]
            
            # Check convergence every 100 steps
            if t % 100 == 0:
                if np.max(np.abs(G[t] - G[t-1])) < tol:
                    G[t:] = G[t-1]  # Fill remaining timesteps with converged distribution
                    break

        # Sum over latter half of timesteps
        half_horizon = horizon // 2
        late_occupancy = G[half_horizon:].sum(axis=0)
        
        return late_occupancy

    def iqltrain(self, next_visit_ids):

        if len(next_visit_ids):
            iql_dataset_indices, zero_indice = self.get_sample_inds(next_visit_ids)

            # Update training indices and visited states
            self.train_inds = np.logical_or(self.train_inds, iql_dataset_indices)
            self.train_inds_list += zero_indice
            self.visited_ids += next_visit_ids

        if self.impute is not None:
            impute_iqldataset = self.get_imputed_dataset(self.train_inds_list, self.impute)
            self.ilagent = None
        else:
            impute_iqldataset = {k: v[self.train_inds_list] for k, v in self.dataset.items()}
            self.ilagent = self.il.train(self.visited_ids)

        q_evaluate = self.qfunction.train(impute_iqldataset)
        self.current_Q = q_evaluate

        if self.impute is not None:
            acc = self.get_impute_acc(q_evaluate, self.bestq)
        else:
            acc = 0.
            
        IQL_agent = offlineRL(q_evaluate, self.ilagent, self.unique_obs, self.total_actions, self.packbits, self.impute, self.s2i)
        
        Js = self.eval_policy(IQL_agent)
        Js = np.mean(Js)

        new_dict = {key: np.max(value) for key, value in self.current_Q.items()}
        state_values = []
        for state in self.unique_obs[self.visited_ids]:
            state_values.append(new_dict[state])
        sorted_state_ids = [item for item, _ in sorted(zip(self.visited_ids, state_values), key=lambda x: x[1], reverse=True)]
        self.visited_ids = sorted_state_ids
        self.state_values = sorted(state_values, reverse=True)
        
        return Js, acc
    
    def get_impute_acc(self, qtable, gt_q):
        def get_argmax_indices(v):
            return set(np.flatnonzero(v == np.max(v)))
        
        preds = {k: get_argmax_indices(v) for k, v in qtable.items()}
        truths = {k: get_argmax_indices(v) for k, v in gt_q.items()}

        # Count as a match if there's any overlap between predicted and true max indices
        matches = sum(preds[k] == truths[k] for k in preds)
        accuracy = matches / len(preds)
        return accuracy
    
    def eval_policy(self, IQL_agent):
        Js = []
        for i in range(self.eval_episodes):
        # for i in trange(self.eval_episodes):
            J = runEpisode(self.env, IQL_agent)
            Js.append(J)
        return np.array(Js)
    
    