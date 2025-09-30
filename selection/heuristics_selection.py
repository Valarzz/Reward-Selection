"""
Heuristics-based state selection methods including:
- uniform: random sampling
- visitation: sample by state visitation frequency
- guided: adaptive method combining visitation heuristics with acquisition functions
Iteratively selects states, trains IQL/offline Q-learning, and evaluates performance.
"""

import numpy as np
import torch
from tqdm import tqdm, trange
import pickle
import os
import time
import copy
from .utils import runEpisode, offlineRL, control_seed, DEFAULT_DEVICE
from .base_selection import base_selection


class heuristics_selection(base_selection):
    def __init__(self,):
        super().__init__()
        self.selection_name = "heuristics"

    def get_state(self, budget):

        sample_size = budget if len(self.visited_ids)==0 else budget - len(self.visited_ids)

        if self.algo == "visit":
            prob = self.norm_prob(copy.deepcopy(self.freq))

        elif self.algo == "uniform":
            prob = self.norm_prob(np.ones(self.total_states) / self.total_states)

        elif self.algo[:4] == "infl":
            if "time" in self.algo.split('-'):
                prob = self.norm_prob(self.discounted_occupancy_time(self.current_Q))
            else:
                prob = self.norm_prob(self.discounted_occupancy(self.current_Q))

            if prob.sum() == 0:
                prob = self.norm_prob(copy.deepcopy(self.freq))

            # if self.algo[-7:] == 'softmax':
            #     prob = self.norm_prob(np.exp(prob / 0.1))

            if self.algo[-6:] == 'argmax':
                idx_desc = np.argsort(prob)[::-1]
                return idx_desc[:sample_size].tolist()

        elif self.algo[:6] == "guided":
            prob, alpha = self.get_q_prob(budget)

            # # softmax or not
            if self.softmax_af:
                prob = np.exp(prob) / np.sum(np.exp(prob))

            prob = self.norm_prob(prob)

        
        prob = prob / prob.sum()
              
        sampled_inds = np.random.choice(self.total_states, p=prob, size=sample_size, replace=False)
        next_visit_ids = sampled_inds.tolist()

        return next_visit_ids  # next unique state id


    def norm_prob(self, prob):
        prob[self.visited_ids] = 0
        prob[-len(self.diff_keys):] = 0
        return prob
    
    def run(self):

        # if self.impute is not None:
        #     self.bestq = self.qfunction.train(self.dataset)

        if self.budget is None:
            total_budget = self.total_states - len(self.diff_keys)

        else:
            if self.budget > self.total_states:
                raise ValueError(f"Budget {self.budget} exceeds the total number of states {self.total_states}.")
            
            else:
                total_budget = self.budget
        
        # if self.algo in ["visit", "uniform"] and self.budget is not None:
        #     next_visit_ids = self.get_state(total_budget)
        #     print(len(next_visit_ids), len(self.train_inds_list), len(self.visited_ids)) 
        #     Js, acc = self.iqltrain(next_visit_ids)
        #     print(len(next_visit_ids), len(self.train_inds_list), len(self.visited_ids)) 
        #     print(Js, acc)
        #     # print(next_visit_ids)
        #     breakpoint()
        #     return


        Js, acc = self.iqltrain([])
        if self.save_result:
            self.save_file(0, Js, acc)
            print(np.mean(Js), 0)

        
        if self.initial_sample:
            Js, acc = self.iqltrain([])
            budget = self.initial_sample
            if self.save_result:
                self.save_file(budget, Js, acc)
                print(np.mean(Js), budget)

        # for budget in trange(self.initial_sample+self.each_query, total_budget+self.each_query, self.each_query):
        for budget in range(self.initial_sample+self.each_query, total_budget+self.each_query, self.each_query):

            budget = min(budget, total_budget)
            
            # this is the only different with all heuristics selection methods
            next_visit_ids = self.get_state(budget)

            Js, acc = self.iqltrain(next_visit_ids)

            if self.save_result:
                self.save_file(budget, Js, acc)
                print(np.mean(Js), acc, budget)
            else:
                # breakpoint()
                print(budget)
                print(len(self.train_inds_list), len(self.visited_ids))
                print(self.visited_ids)
                # print('has repeated? ', len(self.visited_ids) != len(set(self.visited_ids)), len(self.train_inds_list) != len(set(self.train_inds_list)))
                print(Js, acc)

    def save_file(self, budget, Js, acc):

        key = f"{self.algo}-{self.impute_type}-{self.each_query}"

        if self.algo[:6] == "guided":
            key = self.algo + "-" + self.decay + f'-fixed-{self.fixtime}-temp-{self.decay_temp}' + f'-{self.impute_type}-{self.each_query}'

        savepath = f"{self.save_root}/{key}/{budget}/{self.seed}.npy"
        os.makedirs(os.path.dirname(savepath), exist_ok=True)

        np.save(savepath, np.array([Js, acc]))
        # print(savepath)


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
    
    def get_q_prob(self, budget):

        if len(self.visited_ids)==0:
            prob = self.freq
            alpha= 1.
            # sample_size = budget

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
                raise NotImplementedError(f"decay is {self.decay}, it should be chosen from convex, concave, linear")

            alpha = min(1., max(1e-6, alpha))
            # print(alpha)

            dprev = self.afvalue()

            if "infl" in self.algo.split('-'):
                if "time" in self.algo.split('-'):
                    visitation = self.norm_prob(self.discounted_occupancy_time(self.current_Q))
                else:
                    visitation = self.norm_prob(self.discounted_occupancy(self.current_Q))

                if visitation.sum() == 0:
                    visitation = copy.deepcopy(self.freq)

            else:
                visitation = copy.deepcopy(self.freq)

            prob = alpha * visitation + (1 - alpha) * dprev

        return prob, alpha  #, sample_size

    def afvalue(self,):
        dprev = np.zeros(self.total_states)

        if len(self.visited_ids) == 0:
            return dprev

        st = time.time()
        # breakpoint()

        # Compute exponentiated state values for weighting.
        values = np.exp(self.state_values - np.max(self.state_values))
        max_thre = self.af_thre * values.max()

        # iterate from the state with the highest state value
        for sort_id, state_id in enumerate(self.visited_ids):
            good_state = self.unique_obs[state_id]
            good_state = good_state

            # find the indice of states leading to the good state
            if good_state in self.next_state_visitation.keys():
                prime_indices = self.next_state_visitation[good_state]
            else:
                return dprev

            if len(prime_indices):
                # find the unique state of states leading to the good state
                pre_states = self.obs_bits[prime_indices]
                unique_pre_states, counts = np.unique(pre_states, return_counts=True)

                # Find the indices in self.unique_obs corresponding to each unique pre state.
                unique_pre_states_ids = np.array([np.where(self.unique_obs == s)[0][0] for s in unique_pre_states])
                # Filter out states that are already visited.
                mask_not_visited = ~np.isin(unique_pre_states_ids, self.visited_ids)

                # if there exists state that hasn't been visited yet
                if np.any(mask_not_visited):  
                    filtered_ids = unique_pre_states_ids[mask_not_visited]
                    filtered_counts = counts[mask_not_visited]

                    # dprev[filtered_ids] = 1. * values[sort_id]
                    # dprev[filtered_ids] = 1. 

                    dprev[filtered_ids] = filtered_counts * values[sort_id] if self.af_thre else filtered_counts
                    
                    # dprev[filtered_ids] = filtered_counts * values[sort_id]
                    # dprev[filtered_ids] = filtered_counts

                    dprev[self.visited_ids] = 0

                    if (sort_id + 1 == len(values)) or (time.time()-st>150):
                        flag = True
                    else:
                        # flag = (np.count_nonzero(dprev) >= self.each_query) or (values[sort_id+1] < max_thre) 
                        flag = values[sort_id+1] < max_thre if self.af_thre else values[sort_id] != values[sort_id+1]

                        # flag = values[sort_id+1] < max_thre
                        # flag = values[sort_id] != values[sort_id+1]

                    if flag:
                        dprev = dprev / (dprev.sum() + 1e-12)
                        # mask = (dprev > 1e-6)
                        # dprev = np.exp(dprev - np.max(dprev, where=mask, initial=0)) * mask
                        # dprev = dprev / (np.sum(dprev) + 1e-12)
                        return dprev

        dprev = dprev / (dprev.sum() + 1e-12)
        # mask = (dprev > 1e-6)
        # dprev = np.exp(dprev - np.max(dprev, where=mask, initial=0)) * mask
        # dprev = dprev / (dprev.sum() + 1e-12)
        return dprev


