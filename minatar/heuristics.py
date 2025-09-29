import torch
import torch.nn as nn
import numpy as np
from iql import ImplicitQLearning, TwinQ, ValueFunction, plot_line, sample_batch
from bc import BehaviorCloningModel
import pickle
from minatar import Environment
from collections import Counter
from tqdm import tqdm, trange
from copy import deepcopy
from ipdb import launch_ipdb_on_exception
import time
import pandas as pd
import zlib
from collections import defaultdict


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')


class heuristicselection:
    def __init__(self, 
                 iql,
                 bc,
                 dataset, 
                 env, 
                 budget, 
                 each_query,
                 sdim,
                 adim,
                 N=1000, 
                 gamma=0.99, 
                 blind=False, 
                 uniform=False, 
                 decay='linear',
                 zero='vis',
                 fixeddecay=False,
                 fixtime=0.5,
                 train_steps=200_000,
                 eval_period=1_000,
                 batch_size=256,
                 n_eval_episodes=50,
                 decay_temp=6.,
                 initial_sample=1000,
                 game='',
                 ):
        
        self.iql = iql
        self.bc = bc
        self.gamma = gamma
        self.N = N
        # self.budget = budget
        self.each_query = each_query
        self.dataset = dataset
        self.sdim = sdim
        self.adim = adim
        self.env = env
        self.uniform = uniform
        self.decay = decay
        self.zero = zero
        self.fixeddecay = fixeddecay
        self.fixtime = fixtime
        self.blind = blind
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.eval_period = eval_period
        self.n_eval_episodes = n_eval_episodes
        self.decay_temp = decay_temp
        self.initial_sample = initial_sample
        self.game = game

        self.original_bit_count = int(sdim * 100)
        self.extra_bits = (8 - self.original_bit_count % 8) % 8

        self.obs_bits = self.img2byte(dataset['obs'])
        self.obs_prime_bits = self.img2byte(dataset['obs_prime'])

        self.unique_obs, unique_obs_counts = np.unique(self.obs_bits, axis=0, return_counts=True)
        self.freq = unique_obs_counts / unique_obs_counts.sum()
        self.unique_obs_prime = np.unique(self.obs_prime_bits, axis=0)

        self.obs_map = self.bitstobyte(self.obs_bits)
        self.obs_prime_map = self.bitstobyte(self.obs_prime_bits)
        self.unique_obs_map = self.bitstobyte(self.unique_obs)
        self.unique_obs_prime_map = self.bitstobyte(self.unique_obs_prime)

        self.total_states = len(self.unique_obs)
        self.total_sample = len(self.obs_bits)
        self.budget = self.total_states

        # print(self.total_states)
        # print(self.dataset['reward'].min(), self.dataset['reward'].max())
        # breakpoint()

        self.sample_num = len(dataset['obs'])
        self.visited_ids = []
        self.train_inds = np.full((self.total_sample), False, dtype=bool)

        for step in range(train_steps):
            self.bc.train()
            self.bc.update(**self.sample_batch(self.dataset, self.total_sample))
        self.bc.eval()

    def bitstobyte(self, nplist):
        mydict = defaultdict(list)
        for i, row in enumerate(nplist):
            mydict[row.tobytes()].append(i)
        return mydict
    
    def img2byte(self, img):
        data_flat = img.view(img.shape[0], -1)
        data_flat = data_flat.cpu()
        data_np = data_flat.numpy().astype(np.uint8)
        
        data_np = np.pad(data_np, ((0, 0), (0, self.extra_bits)), mode='constant', constant_values=0)
        packed_bits = np.packbits(data_np, axis=1)
        return packed_bits 
    
    def convex_decay(self, rounds, total_rounds=10):
        """Convex decay using a scaled negative exponential function."""
        # scale = np.log(1e5) / (total_rounds - 1)  # Scale to ensure it reaches 0 at total_rounds
        if rounds == self.each_query:
            return 1.
        elif rounds > total_rounds:
            return 0.
        scale = self.decay_temp / (total_rounds - 1)
        return np.exp(-scale * (rounds - 1))
    
    def concave_decay(self, rounds, total_rounds=10):
        if rounds > total_rounds:
            return 0.
        return 1 - self.convex_decay((1+total_rounds - rounds), total_rounds)
    
    def linear_decay(self, rounds, total_rounds=10):
        """Linear decay from 1 to 0."""
        if rounds == self.each_query:
            return 1.
        elif rounds > total_rounds:
            return 0.
        return max(0, 1 - (rounds - 1) / (total_rounds - 1))
    

    def debug_order(self, dist1, dist2):
        df = pd.DataFrame({
            'dist1': dist1,
            'dist2': dist2
        })

        # Rank each distribution from highest to lowest
        # (use method='dense' or 'average' as you prefer for ties)
        df['rank_dist1'] = df['dist1'].rank(method='dense', ascending=False)
        df['rank_dist2'] = df['dist2'].rank(method='dense', ascending=False)

        # If you want to see which elements differ
        df['rank_difference'] = df['rank_dist1'] - df['rank_dist2']
        difffactor = 2

        # For a quick summary:
        num_differences = (abs(df['rank_difference']) > difffactor).sum()
        print(f"Number of elements whose ranks differ: {num_differences}")

        # Filter for rows where the ranks differ
        
        df_different = df[abs(df['rank_difference']) > difffactor]

        # Sort the filtered rows by dist1 (descending)
        df_different_sorted = df_different.sort_values('dist1', ascending=False)

        # Print top rows (e.g., top 10) for inspection
        print(df_different_sorted.head(10))
    

    def get_state(self, budget):

        if self.blind or len(self.visited_ids)==0:
            if self.uniform:
                saf = np.ones(self.total_states) / self.total_states
            else:
                saf = self.freq

            prob = saf

        else:

            totaltozero = int(self.total_states * self.fixtime)
            alpha_idx = budget - self.initial_sample

            if self.decay == 'convex':
                alpha = self.convex_decay(alpha_idx, totaltozero)
            elif self.decay == 'concave':
                alpha = self.concave_decay(alpha_idx, totaltozero)
            elif self.decay == 'linear':
                alpha = self.linear_decay(alpha_idx, totaltozero)
            else:
                alpha = 1.

            alpha = min(1., max(1e-5, alpha))
            # print(alpha)

            dprev = self.afvalue()
            # print(np.nonzero(dprev)[0].shape)
            # breakpoint()
            zero = self.freq if self.zero == 'vis' else np.ones(self.total_states) / self.total_states
            saf = alpha * zero + (1 - alpha) * dprev
            
            # prob = np.exp(saf - np.max(saf))
            prob = saf
            prob[self.visited_ids] = 0
            prob = prob / prob.sum()
        
        if len(self.visited_ids)==0:
            sampled_inds = np.random.choice(len(prob), p=prob, size=budget, replace=False)
            next_visit_ids = sampled_inds.tolist()
            # self.visited_ids = next_visit_ids
        else:
            sample_size = budget - len(self.visited_ids)
            # print(np.count_nonzero(prob), len(self.visited_ids), np.count_nonzero(prob)+len(self.visited_ids), prob.shape, prob.sum())
            sampled_inds = np.random.choice(len(prob), p=prob, size=sample_size, replace=False)
            next_visit_ids = sampled_inds.tolist()
            # self.visited_ids += next_visit_ids

        return next_visit_ids  # next unique state id

    def afvalue(self, debug=False):
        dprev = np.zeros(self.total_states)
        st = time.time()

        values = np.exp(self.state_values - np.max(self.state_values))
        max_thre = 0.9 * values.max()

        if len(self.visited_ids)==0:
            return dprev

        for sort_id, state_id in enumerate(self.visited_ids):
            good_state = self.unique_obs[state_id]

            # prime_indices1 = np.where(np.all(self.obs_prime_bits == good_state, axis=1))[0]
            good_state_bytes = good_state.tobytes()  
            prime_indices = self.obs_prime_map.get(good_state_bytes, [])
            # breakpoint()

            if len(prime_indices):
                pre_states = self.obs_bits[prime_indices]
                unique_pre_states, counts = np.unique(pre_states, axis=0, return_counts=True)

                unique_pre_states_ids = []
                for pre_row in unique_pre_states:
                    pre_row_bytes = pre_row.tobytes()
                    if pre_row_bytes in self.unique_obs_map:
                        unique_pre_states_ids.extend(self.unique_obs_map[pre_row_bytes])
                unique_pre_states_ids = np.array(unique_pre_states_ids, dtype=int)

                filtered_unique_pre_states_ids = unique_pre_states_ids[~np.isin(unique_pre_states_ids, self.visited_ids)]

                if len(filtered_unique_pre_states_ids):  # new ids to be added
                    dprev[unique_pre_states_ids] = counts * values[sort_id]
                    dprev[self.visited_ids] = 0
                    # breakpoint()

                    if (sort_id + 1 == len(values)) or (time.time()-st>150):
                        flag = True
                    else:
                        flag = values[sort_id+1] < max_thre
                        
                    if flag:
                        dprev = dprev / dprev.sum()
                        mask = (dprev > 1e-6)
                        dprev = np.exp(dprev - np.max(dprev, where=mask, initial=0)) * mask
                        dprev = dprev / np.sum(dprev)
                        return dprev
            
        dprev = dprev / (dprev.sum()+1e-12)
        mask = (dprev > 1e-6)
        dprev = np.exp(dprev - np.max(dprev, where=mask, initial=0)) * mask
        dprev = dprev / (dprev.sum()+1e-12)
        return dprev

    def sample_batch(self, dataset, N):  # N = len(dataset['obs'])
        indices = torch.randint(low=0, high=N, size=(self.batch_size,))
        batch = {}
        # print(len(indices), N)
        # breakpoint()
        for key in ['obs', 'action', 'reward', 'obs_prime', 'done']:
            batch[key] = dataset[key][indices].to(DEVICE)
        return batch
    
    def get_env_state(self, s):
        return (torch.tensor(s, device=DEVICE).permute(2, 0, 1)).unsqueeze(0).float()
    
    def evaluate_policy(self, iql, bc, selected_state):
        selected_state_rows = {row.tobytes() for row in selected_state}

        data_return = []
        for _ in range(self.n_eval_episodes):
        # for _ in trange(self.n_eval_episodes):
            with torch.no_grad():
                G = 0.0
                # steps = 0
                # pbar = tqdm(total=2501)

                self.env.reset()
                s = self.get_env_state(self.env.state())
                is_terminated = False
                while(not is_terminated):
                        
                    # Generate data
                    s_bits = np.packbits(s.reshape(s.shape[0], -1).cpu().numpy().astype(np.uint8), axis=1)
                    action = iql(s, 0)

                    reward, terminated = self.env.act(action)
                    s_prime = self.get_env_state(self.env.state())
                    reward = torch.tensor([[reward]], device=DEVICE).float()
                    is_terminated = torch.tensor([[terminated]], device=DEVICE)

                    G += reward.item()
                    # steps += 1
                    # pbar.update(1)

                    s = s_prime

                data_return.append(G)
                # print(G, steps)

        return data_return

    def iqltrain(self, next_visit_ids, test=False):
        
        selected_state_bits = self.unique_obs[next_visit_ids]

        iql_dataset_indices = self.find_matches_in_chunks(selected_state_bits, self.obs_bits)
        # breakpoint()

        self.train_inds = np.logical_or(self.train_inds, iql_dataset_indices)
        impute_inds = np.logical_not(self.train_inds)

        self.visited_ids += next_visit_ids
        selected_state_bits = self.unique_obs[self.visited_ids]
        # print(len(self.visited_ids), self.train_inds.sum())

        iql = deepcopy(self.iql)

        if not test:
            train_steps = self.train_steps
        else:
            if self.game == 'breakout':
                train_steps = 125_000
            elif self.game == 'freeway':
                train_steps = 25_000
            elif self.game == 'seaquest':
                train_steps = 125_000
            elif self.game == 'asterix':
                train_steps = 75_000
            else:
                raise NotImplementedError(f"{self.game} has not been implemented yet.")

        iqldataset = deepcopy(self.dataset)
        iqldataset['reward'][impute_inds] = 0.
        
        iqlsamples = len(iqldataset['obs'])
        # print(iqlsamples)

        # for step in range(train_steps):
        for step in trange(train_steps):

            iql.train()
            iql.update(**self.sample_batch(iqldataset, iqlsamples))
        
        if test:
            iql.eval()
            # bc.eval()
            J = self.evaluate_policy(iql, self.bc, selected_state_bits)
            # allJ.append(J)
        else:
            J = -1.
        
        selected_state = np.unpackbits(selected_state_bits, axis=1, count=self.original_bit_count).reshape(-1, self.sdim, 10, 10)
        state_values = iql.vf(torch.from_numpy(selected_state).float().to(DEVICE)).flatten().cpu().detach().tolist()
        sorted_state_ids = [item for item, _ in sorted(zip(self.visited_ids, state_values), key=lambda x: x[1], reverse=True)]
        policy = None
        self.visited_ids = sorted_state_ids
        self.state_values = sorted(state_values, reverse=True)
        # breakpoint()
        
        return J, policy
    
    def find_matches_in_chunks1(self, selected_state_bits, chunk_size=100):
        results = []

        for start in trange(0, self.total_sample, chunk_size):
        # for start in range(0, self.total_sample, chunk_size):

            end = min(start + chunk_size, self.total_sample)
            chunk = self.obs_bits[start:end]   
            matches_chunk = np.any(
                np.all(chunk[:, None, :] == selected_state_bits[None, :, :], axis=2),
                axis=1
            )
            results.append(matches_chunk)

        return np.concatenate(results, axis=0)

    def find_matches_in_chunks(self, selected_state_bits, target_set, chunk_size=100_000):
        # Convert arrays to hashes
        def hash_rows(arr):
            return np.array([zlib.crc32(row.tobytes()) for row in arr], dtype=np.uint32)

        selected_hashes = set(hash_rows(selected_state_bits))

        results = []
        total_sample = len(target_set)

        # for start in trange(0, total_sample, chunk_size):
        for start in range(0, total_sample, chunk_size):
            end = min(start + chunk_size, total_sample)
            chunk = target_set[start:end]  # self.obs_bits[start:end]

            chunk_hashes = hash_rows(chunk)
            matches_chunk = np.isin(chunk_hashes, list(selected_hashes))
            results.append(matches_chunk)

        return np.concatenate(results, axis=0)


