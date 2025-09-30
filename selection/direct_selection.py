"""
Direct selection method that trains IQL on a specific set of state IDs.
Takes pre-specified visit_ids (provided via config), trains on those states,
and saves results. Used for evaluating specific state selections.
"""

import os
import numpy as np
import hashlib
from .base_selection import base_selection

class DirectSelection(base_selection):
    def __init__(self):
        super().__init__()
        self.selection_name = "direct"

    def _hash_visit_id(self, visit_id):
        """Create a short hash of the visit_id for filename"""
        return hashlib.md5(str(visit_id).encode()).hexdigest()[:12]

    def run(self):
        # Convert visit_ids string to list of integers
        if '_' in self.visit_ids:
            visit_ids = [int(x) for x in self.visit_ids.split('_')]
        else:
            visit_ids = [int(self.visit_ids)]

        # Run IQL training with these visit IDs
        self.direct_budget = len(visit_ids)
        
        # print(visit_ids)
        # visit_ids = [self.s2i[visit_id] for visit_id in visit_ids]
        visit_ids = [self.s2i[visit_id] for visit_id in visit_ids if visit_id in self.s2i]
        # breakpoint()

        Js = []
        acc = []
        for _ in range(3):
            self.train_inds_list = []
            self.visited_ids = []
            J, a = self.iqltrain(visit_ids)
            # print(len(self.train_inds_list), len(self.visited_ids))
            Js.append(J)
            acc.append(a)
        # print(Js)
        # Js = np.mean(Js)
        acc = np.mean(acc)

        self.save_direct_result(Js, acc)

        # J, acc = self.iqltrain(visit_ids)
        # self.save_direct_result(J, acc)

    def save_direct_result(self, Js, acc):
        # Hash the visit IDs for filename
        file_id = self._hash_visit_id(self.visit_ids)
        
        # Create the directory path
        if self.search == "evolutionary":   
            extra = f"_{self.evo_num}_{self.evo_iter}_{self.evo_estimate}"
        else:
            extra = ""
        save_dir = f"{self.root}/{self.env.name}/result/direct/{self.expname}_{self.search}{extra}/{self.direct_budget}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Create the full file path
        dc_ratio = "" if self.dc_ratio is None else f"_{self.dc_ratio}"
        savepath = f"{save_dir}/{file_id}{dc_ratio}.npy"
        
        # Save the numpy array
        # np.save(savepath, [J, acc]) 
        np.save(savepath, [np.mean(Js), acc]) 
        # breakpoint()
        print(savepath)
