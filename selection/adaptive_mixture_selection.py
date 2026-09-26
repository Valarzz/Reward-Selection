"""
Adaptive-mixture selection.

Closed-loop selection that samples the next state to label from a mixture of
an explore distribution and an exploit distribution:

    p(s) = w * p_explore(s) + (1 - w) * p_exploit(s)

- p_exploit: state visitation d(s), either the dataset (behavior) visitation
  frequency or the on-policy discounted occupancy of the current greedy policy.
- p_explore: inverse visitation 1 / (d(s) + eps), or uniform over queryable states.
- w = exp(-|J_true - J_proxy| / exp_scale), recomputed after every query, where
  J_true is the discounted rollout return of the current policy and
  J_proxy = E_{s~d0}[max_a Q(s,a)] is the value predicted by the partially
  labeled Q-function. A small gap means the Q-function is consistent with the
  true return, so we explore more; a large gap shifts mass to exploitation.

Saved results use undiscounted returns (via base_selection.eval_policy) for a
fair comparison with other methods; the gate uses discounted returns so that
J_true and J_proxy are on the same scale.
"""

import os
import numpy as np
from .base_selection import base_selection
from .utils import offlineRL


class AdaptiveMixtureSelection(base_selection):
    def __init__(
        self,
        visitation_source="behavior",
        inverse_visitation_eps=1e-6,
        exp_scale=0.5,
        explore_mode="inv_freq",
    ):
        super().__init__()
        self.selection_name = "adaptive_mixture"
        self.visitation_source = str(visitation_source)
        self.inverse_visitation_eps = float(inverse_visitation_eps)
        self.exp_scale = float(exp_scale)
        self.explore_mode = str(explore_mode)

    def q_proxy_return(self):
        """J_proxy = E_{s~d0}[max_a Q(s,a)] under the current (partially labeled) Q."""
        v = np.zeros(self.total_states, dtype=np.float64)
        for i in range(self.total_states):
            s = self.i2s[i]
            if s in self.current_Q:
                v[i] = float(np.max(self.current_Q[s]))
        return float(np.dot(self.d0, v))

    def discounted_J_true(self):
        """Discounted rollout return of the current offline RL agent (used only for the gate)."""
        agent = offlineRL(
            self.current_Q, self.ilagent, self.unique_obs,
            self.total_actions, self.packbits, self.impute, self.s2i,
        )
        Js = []
        for _ in range(self.eval_episodes):
            G = 0.0
            discount = 1.0
            steps = 0
            state, _ = self.env.reset()
            done = False
            while (not done) and (steps < self.env.MAX_STEPS):
                steps += 1
                action = agent.policy(state)
                state, reward, done, _ = self.env.step(action)
                G += discount * float(reward)
                discount *= self.gamma
            Js.append(G)
        return float(np.mean(Js))

    def delta_gate(self, delta):
        """Exploration weight w = exp(-|delta| / exp_scale) in [0, 1]."""
        delta = abs(float(delta))
        scale = max(1e-8, self.exp_scale)
        return float(np.exp(np.clip(-delta / scale, -60.0, 0.0)))

    def update_gate(self):
        J_true_disc = self.discounted_J_true()
        J_proxy = self.q_proxy_return()
        delta = J_true_disc - J_proxy
        return J_proxy, delta, self.delta_gate(delta)

    def mask_unqueryable(self, prob):
        # already-queried states and states that only appear as next-states cannot be queried
        prob = prob.astype(np.float64, copy=True)
        if len(self.visited_ids):
            prob[self.visited_ids] = 0.0
        if len(self.diff_keys):
            prob[-len(self.diff_keys):] = 0.0
        return prob

    def exploit_prob(self):
        source = self.visitation_source.lower()
        if source == "behavior":
            vis = self.freq.copy()
        elif source == "on_policy":
            vis = self.discounted_occupancy(self.current_Q)
        else:
            raise ValueError(f"Unknown visitation_source={self.visitation_source!r}; expected 'behavior' or 'on_policy'.")

        vis = self.mask_unqueryable(vis)
        if not np.isfinite(vis).all() or vis.sum() <= 0:
            vis = self.mask_unqueryable(self.freq.copy())
        return vis / (vis.sum() + 1e-12)

    def explore_prob(self, visitation_prob):
        mode = self.explore_mode.lower()
        if mode == "inv_freq":
            eps = max(1e-12, self.inverse_visitation_eps)
            inv = 1.0 / (visitation_prob + eps)
        elif mode == "uniform":
            inv = np.ones(self.total_states, dtype=np.float64)
        else:
            raise ValueError(f"Unknown explore_mode={self.explore_mode!r}; expected 'inv_freq' or 'uniform'.")

        inv = self.mask_unqueryable(inv)
        if not np.isfinite(inv).all() or inv.sum() <= 0:
            inv = self.mask_unqueryable(np.ones(self.total_states, dtype=np.float64))
        return inv / (inv.sum() + 1e-12)

    def run(self):
        if self.packbits:
            raise NotImplementedError("Adaptive-mixture selection currently supports tabular domains only (packbits=False).")
        if self.each_query != 1:
            raise ValueError("Adaptive-mixture selection queries one state at a time; set domain.exp.each_query=1.")

        appeared_state_num = self.total_states - len(self.diff_keys)
        if self.budget is None:
            total_budget = appeared_state_num
        else:
            if self.budget > appeared_state_num:
                raise ValueError(f"Budget {self.budget} exceeds the number of queryable states {appeared_state_num}.")
            total_budget = self.budget

        Js, acc = self.iqltrain([])
        J_proxy, delta, w = self.update_gate()
        if self.save_result:
            self.save_file(0, Js, acc)
        print(f"budget {0}: J {np.mean(Js):.3f}, J_proxy {J_proxy:.3f}, delta {delta:.3f}, w {w:.3f}")

        if self.initial_sample:
            # warmup samples are already applied in init_exp
            Js, acc = self.iqltrain([])
            J_proxy, delta, w = self.update_gate()
            budget = self.initial_sample
            if self.save_result:
                self.save_file(budget, Js, acc)
            print(f"budget {budget}: J {np.mean(Js):.3f}, J_proxy {J_proxy:.3f}, delta {delta:.3f}, w {w:.3f}")

        for budget in range(self.initial_sample + 1, total_budget + 1):
            if len(self.visited_ids) >= appeared_state_num:
                break

            p_exploit = self.exploit_prob()
            p_explore = self.explore_prob(p_exploit)
            p_mix = self.mask_unqueryable(w * p_explore + (1.0 - w) * p_exploit)
            total = p_mix.sum()
            if not np.isfinite(total) or total <= 0:
                break
            next_state_id = int(np.random.choice(self.total_states, p=p_mix / total))

            Js, acc = self.iqltrain([next_state_id])
            J_proxy, delta, w = self.update_gate()
            if self.save_result:
                self.save_file(budget, Js, acc)
            print(f"budget {budget}: state {self.i2s[next_state_id]}, J {np.mean(Js):.3f}, "
                  f"J_proxy {J_proxy:.3f}, delta {delta:.3f}, w {w:.3f}")

    def save_file(self, budget, Js, acc):
        key = (f"{self.visitation_source}-{self.explore_mode}-eps{self.inverse_visitation_eps}"
               f"-expscale{self.exp_scale}-{self.impute_type}-{self.each_query}")
        savepath = f"{self.save_root}/{key}/{budget}/{self.seed}.npy"
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        np.save(savepath, np.array([Js, acc]))
