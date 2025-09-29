import os
import numpy as np
import hashlib
from .base_selection import base_selection
import pickle
import time
import subprocess


class EvalSelection(base_selection):
    def __init__(self):
        super().__init__()
        self.selection_name = "eval"

    def _hash_visit_id(self, visit_id):
        """Create a short hash of the visit_id for filename"""
        return hashlib.md5(visit_id.encode()).hexdigest()[:12]

    def run(self):

        self.pri_savepath = f"{self.root}/{self.env.name}/result/training_phase/{self.expname}/best_{self.impute_type}_{self.search}"

        if self.search == "evolutionary":
            extra = f"_{self.evo_num}_{self.evo_iter}_{self.evo_estimate}"
            with open(self.pri_savepath+f"{extra}_.pkl", "rb") as f:
                self.training_phase_dict = pickle.load(f)

        else:
            with open(self.pri_savepath+f"_.pkl", "rb") as f:
                self.training_phase_dict = pickle.load(f)

        print(self.training_phase_dict)

        visit_ids = []
        for budget, save_dict in self.training_phase_dict.items():
            if budget == 0:
                continue
            
            # if self.search == "evolutionary":
            #     visit_ids.append(list(save_dict.keys())[0])
                
            # else:
            #     visit_id = [int(x) for x in list(save_dict.keys())[0].split('_')]
            #     visit_id = [self.i2s[visit_i] for visit_i in visit_id if visit_i in self.i2s]  
            #     visit_id = '_'.join([str(x) for x in visit_id])
            #     visit_ids.append(visit_id)
            visit_ids.append(list(save_dict.keys())[0])
                
        self.push_single_job(visit_ids, self.test_ratios)

        Js = {}
        for vid in visit_ids:
            budget = len(vid.split('_'))
            hashed_id = self._hash_visit_id(vid)
            Js_temp = []
            
            for ratio in self.test_ratios:
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
                    raise FileNotFoundError
                    # pass    

            # breakpoint()
            Js[budget] = {vid: Js_temp}
            # Js[budget] = {vid: np.mean(Js_temp)}
            # if len(Js_temp) > 0:
            #     Js[budget] = np.mean(Js_temp)
            # else:
            #     Js[budget] = -np.inf
        # breakpoint()
        print(Js)

        self.eva_savepath = f"{self.root}/{self.env.name}/result/eval/{self.expname}"
        os.makedirs(self.eva_savepath, exist_ok=True)
        if self.search == "evolutionary":   
            extra = f"_{self.evo_num}_{self.evo_iter}_{self.evo_estimate}"
        else:
            extra = ""
        with open(f"{self.eva_savepath}/best_{self.impute_type}_{self.search}{extra}.pkl", "wb") as f:
            pickle.dump(Js, f)


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
