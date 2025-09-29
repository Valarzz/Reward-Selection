import os
import random
import pickle
import numpy as np
import scipy.signal as signal

class Graph(object):
    def __init__(self,
                 make_pomdp=False,
                 number_of_pomdp_states=2,
                 transitions_deterministic=True,
                 max_length=2,
                 sparse_rewards=False,
                 stochastic_rewards=False,
                 reward='one'):
        """
        We define states in [0 .. 2*max_length - 2].
        That means there are (2*max_length - 1) total states.

        If max_length=8, valid states are [0..14].
        State 13 -> +1 reward, resets to 0, done
        State 14 -> -1 reward, resets to 0, done
        """
        self.reward_type = reward
        self.name = "graph"
        self.MAX_STEPS = 1e5

        # Two actions: 0 or 1
        self.allowable_actions = [0, 1]
        self.n_actions = len(self.allowable_actions)

        # Number of states = 2*max_length - 1
        self.n_dim = 2 * max_length - 1

        self.make_pomdp = make_pomdp
        self.number_of_pomdp_states = number_of_pomdp_states

        # Build a mapping from the environment states -> POMDP states (only matters if make_pomdp=True)
        # We'll split up [1..(2*max_length-2)] among the POMDP states
        # i.e. 1..(14) if max_length=8
        # NB: Just used internally if make_pomdp=True
        split = np.array_split(np.arange(1, 2*max_length-1), number_of_pomdp_states-1)
        self.state_to_pomdp_state = {}
        for pomdp_state, states in enumerate(split):
            for s in states:
                self.state_to_pomdp_state[s] = pomdp_state

        # Force state 0 to map to 0, and the last state (2*max_length-2) to map to (number_of_pomdp_states-1)
        # e.g. for max_length=8 => state 14 => map to pomdp_state=1 if number_of_pomdp_states=2
        self.state_to_pomdp_state[0] = 0
        self.state_to_pomdp_state[2*max_length - 2] = number_of_pomdp_states - 1

        self.transitions_deterministic = transitions_deterministic
        self.slippage = 0.25
        self.max_length = max_length
        self.sparse_rewards = sparse_rewards
        self.stochastic_rewards = stochastic_rewards
        
        # For overwriting rewards or forcing an absorbing state (optional features)
        self.reward_overwrite = None if reward == 'one' else self.generate_random_values()
        self.absorbing_state = None
        self.end_state = 100000

        self.reset()
        _, self.pi_opt = self.value_iteration()
        
        # Initialize
        self.reset()

    def print_graph(self):
        # For clarity, store the original state so we can restore later
        original_state = self.state
        original_done = self.done

        print("Graph of transitions (deterministic) for each (state, action):")
        for s in range(self.num_states()):
            # We'll skip the case s=15 if it's truly terminal in your scenario 
            # but let's just do them all to see what happens
            for a in self.allowable_actions:
                # Set environment to state s
                self.state = s
                self.done = False  # forcibly say it's not done so that step() will proceed
            
                # Step
                next_s, reward, done, info = self.step(a)
                # next_s = next_s.item()  # from array to int
                print(f"  {s} --(a={a}, r={reward})--> {next_s}   done={done}")

        # Restore
        self.state = original_state
        self.done = original_done

    def num_states(self):
        return self.n_dim  # = 2*max_length - 1

    def num_actions(self):
        return self.n_actions

    def reset(self):
        self.state = 0
        self.done = False
        return self.state, None

    def step(self, action):
        """
        Moves from current state -> next state given 'action' in {0,1}.
        Returns (next_state, reward, done, info).
        """
        assert action in self.allowable_actions
        assert not self.done, "Episode is already done! Call reset()."

        # If rewards are stochastic, we add noise later (+/-)
        reward = 0.0 if not self.stochastic_rewards else np.random.randn()
        
        # If using POMDP, we track the 'previous' displayed state
        prev_state_for_reward_key = (self.state_to_pomdp_state[self.state]
                                     if self.make_pomdp else self.state)

        # 1) Check if we are in a terminal
        #    Terminal states: 13 -> +1, 14 -> -1
        if self.state == (2*self.max_length - 3):  # e.g. 13 if max_length=8
            reward = 1.0 if not self.stochastic_rewards else (1.0 + np.random.randn())
            self.state = self.end_state
            self.done = True

        elif self.state == (2*self.max_length - 2):  # e.g. 14 if max_length=8
            reward = -1.0 if not self.stochastic_rewards else (-1.0 + np.random.randn())
            self.state = self.end_state
            self.done = True

        else:
            # 2) Normal transitions
            if self.state == 0:
                # If we are at the start state (0)
                if action == 0:  
                    # action=0 from state=0
                    if self.transitions_deterministic:
                        self.state += 1
                    else:
                        self.state = int(np.random.choice([self.state+1, self.state+2],
                                                          p=[1-self.slippage, self.slippage]))
                else:
                    # action=1 from state=0
                    if self.transitions_deterministic:
                        self.state += 2
                    else:
                        self.state = int(np.random.choice([self.state+2, self.state+1],
                                                          p=[1-self.slippage, self.slippage]))
            else:
                # If we are in some state >=1 and <13
                if action == 0:
                    # action=0
                    if self.transitions_deterministic:
                        if self.state % 2 == 1:  # odd
                            self.state += 2
                        else:                    # even
                            self.state += 1
                    else:
                        # nondeterministic transitions
                        if self.state % 2 == 1:
                            self.state = int(np.random.choice([self.state+2, self.state+3],
                                                              p=[1-self.slippage, self.slippage]))
                        else:
                            self.state = int(np.random.choice([self.state+1, self.state+2],
                                                              p=[1-self.slippage, self.slippage]))
                else:
                    # action=1
                    if self.transitions_deterministic:
                        if self.state % 2 == 1:  # odd
                            self.state += 3
                        else:                    # even
                            self.state += 2
                    else:
                        # nondeterministic transitions
                        if self.state % 2 == 1:
                            self.state = int(np.random.choice([self.state+3, self.state+2],
                                                              p=[1-self.slippage, self.slippage]))
                        else:
                            self.state = int(np.random.choice([self.state+2, self.state+1],
                                                              p=[1-self.slippage, self.slippage]))

            # 3) Dense deterministic/stochastic reward
            #    If not sparse, we do +/- every step
            if not self.sparse_rewards and not self.done:
                if self.state % 2 == 1:  # odd
                    reward = (1.0 if not self.stochastic_rewards
                              else 1.0 + np.random.randn())
                else:                     # even
                    reward = (-1.0 if not self.stochastic_rewards
                              else -1.0 + np.random.randn())

        # 4) If using any "reward_overwrite", apply it
        if self.reward_overwrite is not None:
            # If the episode ended, we treat next_state as "absorbing_state" in the key
            # next_state_for_reward_key = (self.absorbing_state if self.done 
            #                              else self.state)
            # key = (int(prev_state_for_reward_key), int(action), int(next_state_for_reward_key))
            key = int(self.state)
            if key in self.reward_overwrite:
                # Possibly sample from a distribution
                # reward = np.random.choice(list(self.reward_overwrite[key]),
                #                           p=list(self.reward_overwrite[key].values()))
                reward = self.reward_overwrite[key]
            else:
                reward = 0.0

        # 5) Return next_observation (depending on POMDP or not)
        if self.make_pomdp:
            obs = self.state_to_pomdp_state[self.state]
        else:
            obs = self.state
        return obs, reward, self.done, {}

    def render(self, a=None, r=None, return_arr=False):
        """
        Basic text-based visualization. 
        If you want something more, you can track self.state and print it out,
        or do a custom plotting of the transitions.
        """
        # For a simple approach, show start (S) vs. done (D)
        # We'll mark terminal states (13 and 14) if we are in them at the moment.
        start_symbol = "S" if self.state == 0 else " "
        if self.state == (2*self.max_length-3):
            terminal_symbol = "+T"  # positive terminal
        elif self.state == (2*self.max_length-2):
            terminal_symbol = "-T"  # negative terminal
        else:
            terminal_symbol = " "

        print(f"State: {self.state}  [{start_symbol}{terminal_symbol}]")
        if a is not None and r is not None:
            print(f"  (Action={a}, Reward={r})")
        print()

    @staticmethod
    def discounted_sum(costs, discount):
        """
        Utility for summing discounted rewards: sum_{t=0..T-1} [ discount^t * costs[t] ]
        """
        y = signal.lfilter([1], [1, -discount], x=costs[::-1])
        return y[::-1][0]


    def create_dataset(self, data_collecting, state, eps=None):
        p = np.random.random()

        if eps is not None:
            thre = eps
        else:
            if data_collecting == 'good':
                thre = 0.7
            elif data_collecting == 'mid':
                thre = 0.5
            elif data_collecting == 'bad':
                thre = 0.3
            else:
                raise NotImplementedError(f"data_collecting is {data_collecting}, it should be chosen from good, mid, bad")

        # return self.pi_opt[state]
        return self.pi_opt[state] if p < thre else 1 - self.pi_opt[state]

    def generate_random_values(self,):

        random_reward_path = "RLLF/domains/graph_reward.pkl"
        if os.path.exists(random_reward_path):
            with open(random_reward_path, 'rb') as f:
                return pickle.load(f)

        else:
            num_nodes = self.n_dim
            reward = {i: random.randint(-5, 15) for i in range(num_nodes)}
            with open(random_reward_path, 'wb') as f:
                pickle.dump(reward, f)
            return reward

    def build_transition_model(self, ):
        """
        Build a standard MDP transition model:
        P[s][a] = list of (prob, next_s, reward, done).
        Because transitions_deterministic=True, each (s,a) will have exactly
        one outcome with prob=1.0.
        """
        nS = self.num_states()
        nA = len(self.allowable_actions)
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        # Temporarily store original
        original_state = self.state
        original_done = self.done

        for s in range(nS):
            for a in range(nA):
                # Force environment to be at state s, not done
                self.state = s
                self.done = False

                next_s, r, done, _ = self.step(a)
                # next_s = int(next_obs[0])

                # Because this is deterministic, we have exactly one outcome
                P[s][a].append((1.0, next_s, r, done))

        # Restore
        self.state = original_state
        self.done = original_done

        return P

    def value_iteration(self, gamma=0.99, tol=1e-8):
        """
        Standard Value Iteration for an MDP with transition model P.
        
        Args:
        P: dict of dicts
            P[s][a] = list of (prob, next_s, reward, done)
        nS: number of states
        nA: number of actions
        gamma: discount factor
        tol: convergence tolerance

        Returns:
        V:  np.array of shape (nS,)   (optimal value function)
        pi: np.array of shape (nS,)   (optimal deterministic policy)
        """

        nS = self.num_states()       
        nA = len(self.allowable_actions)  
        P = self.build_transition_model()

        V = np.zeros(nS)
        while True:
            delta = 0.0
            for s in range(nS):
                v_old = V[s]
                # Compute Q(s,a) = sum_{next_s} prob*(r + gamma*V[next_s]*(1 - done))
                q_vals = np.zeros(nA)
                for a in range(nA):
                    for (prob, next_s, r, done) in P[s][a]:
                        q_vals[a] += prob * (r + gamma * (0 if done else V[next_s]))
                V[s] = np.max(q_vals)
                delta = max(delta, abs(v_old - V[s]))
            if delta < tol:
                break

        # Extract policy
        policy = np.zeros(nS, dtype=int)
        for s in range(nS):
            q_vals = np.zeros(nA)
            for a in range(nA):
                for (prob, next_s, r, done) in P[s][a]:
                    q_vals[a] += prob * (r + gamma * (0 if done else V[next_s]))
            policy[s] = np.argmax(q_vals)
        return V, policy
