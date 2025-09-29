import numpy as np
import torch
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import itertools
import matplotlib.pyplot as plt
import argparse
import pickle
import os
import subprocess
import multiprocessing as mp
from functools import partial
import time
import ast
import json
import copy
import hashlib
from .utils import runEpisode, offlineRL, control_seed, DEFAULT_DEVICE
from .base_selection import base_selection

class training_phase_selection(base_selection):
    def __init__(self,):
        super().__init__()
        self.selection_name = "training_phase"
    
    def _hash_visit_id(self, visit_id):
        """Create a short hash of the visit_id for filename"""
        return hashlib.md5(visit_id.encode()).hexdigest()[:12]
    
    def run(self, ):
        self.pri_savepath = f"{self.save_root}/best_{self.impute_type}_{self.search}"
        os.makedirs(self.save_root, exist_ok=True)

        appeared_state_num = self.total_states - len(self.diff_keys)
        
        if self.budget is None:
            total_budget = appeared_state_num

        else:
            if self.budget > appeared_state_num:
                raise ValueError(f"Budget {self.budget} exceeds the total number of states {appeared_state_num}.")
            
            else:
                total_budget = self.budget

        self.training_phase_dict = {}
        self.sorted_keys = []

        Js, _ = self.iqltrain([])
        self.training_phase_dict[0] = {0: Js}
        print(0, 'nan', Js)
        # breakpoint()

        each_query = self.each_query if self.search == "evolutionary" else 1

        for budget in range(each_query, total_budget+each_query, each_query):

            budget = min(budget, total_budget)
            self.get_result_on_budget(budget)

    def get_result_on_budget(self, budget):
        if self.search == "evolutionary":
            if self.evo_estimate == "argmax":
                self.push_evolutionary_argmax_job(budget)
            else:
                self.push_evolutionary_sample_job(budget)

        else:
            self.push_sequential_job(budget)

        self.save_pri()

        
    def save_pri(self,):
        ttstr = "tt" if self.tt else ""

        if self.search == "evolutionary":
            extra = f"_{self.evo_num}_{self.evo_iter}_{self.evo_estimate}"
            print(self.pri_savepath+f"{extra}_{ttstr}.pkl")
            with open(self.pri_savepath+f"{extra}_{ttstr}.pkl", "wb") as f:
                pickle.dump(self.training_phase_dict, f)

        else:
            print(self.pri_savepath+f"_{ttstr}.pkl")
            with open(self.pri_savepath+f"_{ttstr}.pkl", "wb") as f:
                pickle.dump(self.training_phase_dict, f)


    def push_sequential_job(self, budget):

        if self.search == "greedy":
            k = self.sorted_keys[0] if len(self.sorted_keys) else False
            visit_ids = self.push_greedy_job(k)
        elif self.search == "beam":
            visit_ids = self.push_beam_job()

        if self.tt:
            self.push_single_job(visit_ids, self.train_ratios)
        else:
            self.push_single_job(visit_ids)
        # breakpoint()

        Js = {}
        for vid in visit_ids:
            hashed_id = self._hash_visit_id(vid)
            Js_temp = []

            if self.tt:
                for ratio in self.train_ratios:
                    loadpath = f"{self.root}/{self.env.name}/result/direct/{self.expname}_{self.search}/{budget}/{hashed_id}_{ratio}.npy"
                    # breakpoint()
                    try:
                        J, a = np.load(loadpath)
                        Js_temp.append(J)
                    except FileNotFoundError:
                        print(f"Warning: Could not find file for visit_id {vid} at {loadpath}")
                        # raise FileNotFoundError
                        pass
            else:
                loadpath = f"{self.root}/{self.env.name}/result/direct/{self.expname}_{self.search}/{budget}/{hashed_id}.npy"
                try:
                    J, a = np.load(loadpath)
                    Js_temp.append(J)
                except FileNotFoundError:
                    print(f"Warning: Could not find file for visit_id {vid} at {loadpath}")
                    raise FileNotFoundError
                    # pass

            if len(Js_temp) > 0:    
                Js[vid] = np.mean(Js_temp)
            else:
                Js[vid] = -np.inf

        self.sorted_keys = sorted(Js, key=Js.get, reverse=True)
        best_key = self.sorted_keys[0]

        if self.tt:
            self.push_single_job([best_key], self.test_ratios)

            hashed_id = self._hash_visit_id(best_key)
            Js_test = []

            for ratio in self.test_ratios:
                loadpath = f"{self.root}/{self.env.name}/result/direct/{self.expname}_{self.search}/{budget}/{hashed_id}_{ratio}.npy"
                try:
                    J, a = np.load(loadpath)
                    Js_test.append(J)
                except FileNotFoundError:
                    print(f"Warning: Could not find file for visit_id {vid} at {loadpath}")
                    # raise FileNotFoundError
                    pass

            Js_test = np.mean(Js_test)
            save_dict = {best_key: [Js[best_key], Js_test]}
            print(budget, best_key, Js[best_key], Js_test)

        else:
            save_dict = {best_key: Js[best_key]}
            print(budget, best_key, Js[best_key])

        self.training_phase_dict[budget] = save_dict
        # breakpoint()
        # return Js

    def push_evolutionary_sample_job(self, budget):
        
        # Parameters for evolutionary search
        N = self.evo_num  # Number of samples to generate
        M = int(0.2 * N)  # Number of top performers to keep
        K = self.evo_iter  # Number of iterations

        dim = len(self.unique_obs_keys_notdone)
        mean = np.zeros(dim)
        std = np.ones(dim)
        elite_num = max(3, M)

        for gen in range(K):
            # Sample M parameter vectors from current Gaussian
            Q = np.random.randn(N, dim) * std + mean
            Q_norm = self.normalize_Q(Q)

            meanJs = np.zeros(N)
            for j in range(5):
                visit_ids = []
                for i in range(N):
                    top_indices = np.random.choice(dim, size=budget, p=Q_norm[i])
                    top_indices = [self.i2s[top_indice] for top_indice in top_indices]
                    visit_id = '_'.join(map(str, sorted(top_indices)))
                    visit_ids.append(visit_id)

                if self.tt:
                    Js = self.get_Js(visit_ids, budget, self.train_ratios)
                else:
                    Js = self.get_Js(visit_ids, budget)
                
                meanJs += Js

            # Js = meanJs.mean(axis=0)
            # breakpoint()
            ind = np.argpartition(-meanJs, elite_num)[:elite_num]
            top_Q = Q[ind]
            mean = top_Q.mean(axis=0)
            std = top_Q.std(axis=0) + 1e-8
        
        # For the final mean, select top 'budget' indices
        # top_indices = np.argsort(mean)[::-1][:budget]
        # top_indices = [self.i2s[top_indice] for top_indice in top_indices]
        # visit_id = '_'.join(map(str, sorted(top_indices)))
        visit_ids = []
        for i in range(100):
            top_indices = np.random.choice(dim, size=budget, p=self.normalize_Q(mean))
            top_indices = [self.i2s[top_indice] for top_indice in top_indices]
            visit_id = '_'.join(map(str, sorted(top_indices)))
            visit_ids.append(visit_id)
        
        # breakpoint()
        if self.tt:
            Js = self.get_Js(visit_ids, budget, self.train_ratios)
            Js_test = self.get_Js(visit_ids, budget, self.test_ratios)
            save_dict = {visit_id: [Js, Js_test]}
            print(budget, visit_id, Js, Js_test)
        else:
            Js = self.get_Js(visit_ids, budget)
            save_dict = {visit_id: Js}
            print(budget, visit_id, Js)  #

        self.training_phase_dict[budget] = save_dict
    
    def push_evolutionary_argmax_job(self, budget):
        
        # Parameters for evolutionary search
        N = self.evo_num  # Number of samples to generate
        M = int(0.2 * N)  # Number of top performers to keep
        K = self.evo_iter  # Number of iterations

        dim = len(self.unique_obs_keys_notdone)
        mean = np.zeros(dim)
        std = np.ones(dim)
        elite_num = max(3, M)

        for gen in range(K):
            # Sample M parameter vectors from current Gaussian
            Q = np.random.randn(N, dim) * std + mean
            # Q_norm = self.normalize_Q(Q)

            visit_ids = []
            for i in range(N):
                # visit_id = '_'.join(map(str, sorted(np.random.choice(dim, size=budget, p=Q_norm[i]))))
                # Select top 'budget' indices for each Q[i]
                top_indices = np.argsort(Q[i])[::-1][:budget]
                top_indices = [self.i2s[top_indice] for top_indice in top_indices]
                visit_id = '_'.join(map(str, sorted(top_indices)))
                visit_ids.append(visit_id)

            if self.tt:
                Js = self.get_Js(visit_ids, budget, self.train_ratios)
            else:
                Js = self.get_Js(visit_ids, budget)

            ind = np.argpartition(-Js, elite_num)[:elite_num]
            top_Q = Q[ind]
            mean = top_Q.mean(axis=0)
            std = top_Q.std(axis=0) + 1e-8
            
        # visit_ids = []
        # For the final mean, select top 'budget' indices
        top_indices = np.argsort(mean)[::-1][:budget]
        top_indices = [self.i2s[top_indice] for top_indice in top_indices]
        visit_id = '_'.join(map(str, sorted(top_indices)))
        # for i in range(100):
        #     visit_ids.append(visit_id)
        
        # breakpoint()
        if self.tt:
            Js = self.get_Js([visit_id], budget, self.train_ratios)
            Js_test = self.get_Js([visit_id], budget, self.test_ratios)
            save_dict = {visit_id: [Js, Js_test]}
            print(budget, visit_id, Js, Js_test)
        else:
            Js = self.get_Js([visit_id], budget)
            save_dict = {visit_id: Js}
            print(budget, visit_id, Js)

        self.training_phase_dict[budget] = save_dict

    def get_Js(self, visit_ids, budget, ratios=[]):

        self.push_single_job(visit_ids, ratios)
        
        Js = np.zeros(len(visit_ids))
        for j, vid in enumerate(visit_ids):
            hashed_id = self._hash_visit_id(vid)
            Js_temp = []

            if self.tt:
                for ratio in ratios:
                    if self.search == "evolutionary":   
                        extra = f"_{self.evo_num}_{self.evo_iter}_{self.evo_estimate}"
                    else:
                        extra = ""
                    loadpath = f"{self.root}/{self.env.name}/result/direct/{self.expname}_{self.search}{extra}/{budget}/{hashed_id}_{ratio}.npy"
                    # breakpoint()
                    try:
                        J, a = np.load(loadpath)
                        Js_temp.append(J)
                    except FileNotFoundError:
                        print(f"Warning: Could not find file for visit_id {vid} at {loadpath}")
                        # raise FileNotFoundError
                        pass    
            else:
                if self.search == "evolutionary":
                    extra = f"_{self.evo_num}_{self.evo_iter}_{self.evo_estimate}"
                else:
                    extra = ""
                loadpath = f"{self.root}/{self.env.name}/result/direct/{self.expname}_{self.search}{extra}/{budget}/{hashed_id}.npy"
                try:
                    J, a = np.load(loadpath)
                    Js_temp.append(J)
                except FileNotFoundError:
                    print(f"Warning: Could not find file for visit_id {vid} at {loadpath}")
                    raise FileNotFoundError
                    # pass

            if len(Js_temp) > 0:
                Js[j] = np.mean(Js_temp)
            else:
                Js[j] = -np.inf
        return Js


    def normalize_Q(self, Q):
        Q_min = Q.min(axis=1, keepdims=True)
        Q_max = Q.max(axis=1, keepdims=True)
        Q_norm = (Q - Q_min) / (Q_max - Q_min + 1e-8)
        Q_norm = Q_norm / Q_norm.sum(axis=1, keepdims=True)
        return Q_norm
    
    def push_beam_job(self, ):
        new_ids = []

        if len(self.sorted_keys) == 0:
            return self.push_greedy_job(False)

        else:
            for i in range(self.beam_size):
                if i >= len(self.sorted_keys):
                    break
                k = self.sorted_keys[i] 
                new_ids += self.push_greedy_job(k)
                
            unique = set()
            for s in new_ids:
                parts = s.split("_")
                parts = [int(p) for p in parts]
                parts.sort()
                parts = [str(p) for p in parts]
                normalized = "_".join(parts)
                unique.add(normalized)
            # breakpoint()
            return list(unique)

    def push_greedy_job(self, k):

        used = set(map(int, k.split("_"))) if k else set()

        new_ids = []
        # breakpoint()
        for s in self.unique_obs_keys_notdone:
            if s not in used:
                # If k is non-empty, append "_s", else just str(s)
                if k:
                    new_ids.append(k + "_" + str(s))
                else:
                    new_ids.append(str(s))
        return new_ids

        # new_ids = []
        # # breakpoint()
        # for s_id in range(len(self.unique_obs_keys_notdone)):
        #     if s_id not in used:
        #         # If k is non-empty, append "_s", else just str(s)
        #         if k:
        #             new_ids.append(k + "_" + str(s_id))
        #         else:
        #             new_ids.append(str(s_id))
        # return new_ids

    def push_single_job(self, visit_ids, ratios=[]):
        ROOT = "YOUR_ROOT"
        job_ids = []

        if len(ratios) == 0:
            # Submit a job array over visit_ids (no dataset.dc_ratio)
            fn = f"{ROOT}/input/direct_{self.expname}_{self.impute_type}_{self.search}_{np.random.randint(1000000)}.sb"
            if os.path.isfile(fn):
                try:
                    os.remove(fn)
                except FileNotFoundError:
                    pass

            lines = []
            lines.append(f'#!/bin/bash \n')
            lines.append(f'#SBATCH --job-name=direct_{self.expname}_{self.impute_type}_{self.search} \n')
            lines.append(f'#SBATCH --nodes=1 \n')
            lines.append(f'#SBATCH --ntasks=1  \n')
            lines.append(f'#SBATCH --cpus-per-task=2 \n')
            lines.append(f'#SBATCH --mem=300M \n')
            lines.append(f'#SBATCH --time=00:10:00 \n')
            lines.append(f'#SBATCH --array=0-{len(visit_ids)-1} \n')
            lines.append(f'#SBATCH -o {ROOT}/output/direct_temp.out \n')
            lines.append('source conda.sh \n')
            lines.append('conda activate icu \n')
            lines.append('cd YOUR_ROOT/RLLF \n')
            lines.append('VISIT_IDS_LIST=(' + ' '.join(f'"{vid}"' for vid in visit_ids) + ')\n')
            lines.append('VISIT_ID=${VISIT_IDS_LIST[$SLURM_ARRAY_TASK_ID]}\n')
            lines.append('HASHED_ID=$(echo -n "$VISIT_ID" | md5sum | cut -c1-12)\n')
            expr = f'srun python main.py selection=direct selection_params.search={self.search} selection_params.evo_num={self.evo_num} selection_params.evo_iter={self.evo_iter} selection_params.evo_estimate={self.evo_estimate} general.expname={self.expname} general.seed={self.seed} domain={self.env.name} domain.domain.reward={self.reward_type}  dataset.data_collecting=good dataset.dataset_seed={self.dataset_seed} domain.exp.impute={self.impute_type} selection_params.visit_ids="\'$VISIT_ID\'" selection_params.hashed_id="\'$HASHED_ID\'" '
            lines.append(f'{expr} \n')
            # print(expr)
            with open(fn, 'w') as f:
                for line in lines:
                    f.write(f'{line}\n')
            submit_cmd = ["sbatch", fn]
            result = subprocess.run(submit_cmd, capture_output=True, text=True)
            output = result.stdout.strip()
            if result.returncode != 0:
                raise RuntimeError(f"Failed to submit job:\n{result.stderr}")
            try:
                job_id_str = output.split()[-1]
                job_id = int(job_id_str)
            except (IndexError, ValueError):
                raise RuntimeError(f"Could not parse job ID from sbatch output: {output}")
            job_ids.append(job_id)
            if os.path.isfile(fn):
                try:
                    os.remove(fn)
                except FileNotFoundError:
                    pass

        else:
            # For each visit_id, submit a job array over self.train_ratios
            for vid in visit_ids:
                fn = f"{ROOT}/input/direct_{self.expname}_{self.impute_type}_{self.search}_{np.random.randint(1000000)}.sb"
                if os.path.isfile(fn):
                    try:
                        os.remove(fn)
                    except FileNotFoundError:
                        pass
                lines = []
                lines.append(f'#!/bin/bash \n')
                lines.append(f'#SBATCH --job-name=direct_{self.expname}_{self.impute_type}_{self.search} \n')
                lines.append(f'#SBATCH --nodes=1 \n')
                lines.append(f'#SBATCH --ntasks=1  \n')
                lines.append(f'#SBATCH --cpus-per-task=2 \n')
                lines.append(f'#SBATCH --mem=300M \n')
                lines.append(f'#SBATCH --time=00:10:00 \n')
                lines.append(f'#SBATCH --array=0-{len(self.train_ratios)-1} \n')
                lines.append(f'#SBATCH -o {ROOT}/output/direct_temp.out \n')
                lines.append('source conda.sh \n')
                lines.append('conda activate icu \n')
                lines.append('cd YOUR_ROOT/RLLF \n')
                # Prepare the ratios as a bash array
                lines.append('RATIOS_LIST=(' + ' '.join(str(r) for r in ratios) + ')\n')
                lines.append('DS_RATIO=${RATIOS_LIST[$SLURM_ARRAY_TASK_ID]}\n')
                lines.append(f'VISIT_ID="{vid}"\n')
                lines.append('HASHED_ID=$(echo -n "$VISIT_ID" | md5sum | cut -c1-12)\n')
                expr = (
                    f'srun python main.py dataset.dc_ratio=$DS_RATIO selection=direct selection_params.search={self.search} selection_params.evo_num={self.evo_num} selection_params.evo_iter={self.evo_iter} selection_params.evo_estimate={self.evo_estimate} '
                    f'general.expname={self.expname} general.seed={self.seed} domain={self.env.name} '
                    f'domain.domain.reward={self.reward_type} dataset.data_collecting=good dataset.dataset_seed={self.dataset_seed} '
                    f'domain.exp.impute={self.impute_type} selection_params.visit_ids="\'$VISIT_ID\'" selection_params.hashed_id="\'$HASHED_ID\'"'
                )
                lines.append(f'{expr} \n')
                # print(expr)
                with open(fn, 'w') as f:
                    for line in lines:
                        f.write(f'{line}\n')
                submit_cmd = ["sbatch", fn]
                result = subprocess.run(submit_cmd, capture_output=True, text=True)
                output = result.stdout.strip()
                if result.returncode != 0:
                    raise RuntimeError(f"Failed to submit job:\n{result.stderr}")
                try:
                    job_id_str = output.split()[-1]
                    job_id = int(job_id_str)
                except (IndexError, ValueError):
                    raise RuntimeError(f"Could not parse job ID from sbatch output: {output}")
                job_ids.append(job_id)
                if os.path.isfile(fn):
                    try:
                        os.remove(fn)
                    except FileNotFoundError:
                        pass
                # print(fn)
                # breakpoint()

        # Wait for all jobs to finish
        while job_ids:
            time.sleep(5)
            # Check all job_ids
            still_running = []
            for job_id in job_ids:
                check_cmd = ["squeue", "-j", str(job_id)]
                check_result = subprocess.run(check_cmd, capture_output=True, text=True)
                if str(job_id) in check_result.stdout:
                    still_running.append(job_id)
            job_ids = still_running
        return
