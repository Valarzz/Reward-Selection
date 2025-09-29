import numpy as np
import gymnasium as gym
from typing import Tuple, Union


class ToyText:
    def __init__(self, envname, reward=None):
        self.reward_type = reward
        
        if envname == 'frozenlake':
            self.env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
            self.n_states = 16  # 4x4 grid
            self.n_actions = 4  # LEFT, DOWN, RIGHT, UP
            self.MAX_STEPS = int(1e3)
            
        elif envname == 'cliffwalk':
            self.env = gym.make('CliffWalking-v0')
            self.n_states = 48  # 4x12 grid
            self.n_actions = 4  # UP, RIGHT, DOWN, LEFT
            self.MAX_STEPS = int(100)

        elif envname == 'blackjack':
            self.env = gym.make('Blackjack-v1')
            # self.n_states = 704  
            self.n_states = 360  
            self.n_actions = 2  
            self.MAX_STEPS = int(1e3)

        else:
            raise NotImplementedError(f"envname is {envname}, it should be chosen from frozenlake, cliffwalk, blackjack")
        
        self.name = envname
        self.envname = envname
        
    def create_dataset(self, data_collecting, s, eps=None):
        # Set epsilon based on data quality desired
        if eps is not None:
            epsilon = eps
        else:
            if data_collecting == 'good':
                epsilon = 0.5
            elif data_collecting == 'mid':
                return self.env.action_space.sample()
            elif data_collecting == 'bad':
                epsilon = 0.1

        # epsilon = 1.
        if np.random.random() < epsilon:
            # Return optimal action with probability epsilon
            if self.envname == 'frozenlake':
                # Optimal policy dictionary for 4x4 FrozenLake
                # Actions: LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3
                optimal_policy = {
                    0: [1, 2],      # Start: can go DOWN or RIGHT
                    1: [2],      # Can go DOWN or RIGHT
                    2: [1],         # Must go RIGHT (avoid hole at 5)
                    3: [0],         # Must go DOWN (avoid hole at 7)
                    4: [1],      # Can go DOWN or RIGHT
                    6: [1],      # Can go DOWN or RIGHT
                    8: [2],         # Must go RIGHT (avoid hole at bottom)
                    9: [1, 2],         # Must go RIGHT
                    10: [1],        # Must go RIGHT (avoid hole)
                    13: [2],        # Must go RIGHT to goal
                    14: [2],     # Can go DOWN or RIGHT to goal
                }
                # If state has optimal actions, randomly choose one, else random action
                if s in optimal_policy:
                    a = np.random.choice(optimal_policy[s])
                    # print(s, a)
                    return a
                    
            elif self.envname == 'cliffwalk':
                # Convert state number to row, col coordinates
                row = s // 12  # 4x12 grid
                col = s % 12
                
                # Start is at (3,0), Goal is at (3,11)
                # For any state, move towards top row first, then right, then down to goal
                if row in [0, 1] and col < 11:
                    return np.random.choice([1, 2])
                elif row == 2 and col < 11:
                    return 1
                elif row == 3 and col == 0:
                    return 0
                elif col == 11:
                    return 2

            elif self.envname == 'blackjack':
                return self.black_jack_optimal_policy(s)
                
        else:
            # Return random action with probability 1-epsilon
            return self.env.action_space.sample()
    
    def step(self, action):
        try:
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)  # Explicitly convert to Python bool
            return self.s2i(next_state), reward, done, info
        except Exception as e:
            # Fallback for older gym versions
            next_state, reward, done, info = self.env.step(action)
            return self.s2i(next_state), reward, bool(done), info

    def num_states(self):
        return self.n_states

    def num_actions(self):
        return self.n_actions

    def reset(self):
        try:
            state, info = self.env.reset()
            return self.s2i(state), info
        except Exception as e:
            # Fallback for older gym versions
            return self.s2i(self.env.reset()), None

    # --------------------------------------------------------------------- #
    # Basic‑strategy expert
    # --------------------------------------------------------------------- #
    def black_jack_optimal_policy(self, s_idx: int) -> int:
        player, dealer, soft = self.i2s(s_idx)

        # basic strategy (infinite deck, dealer stands on soft 17)
        if soft:                       # ----- soft hand -----
            if player >= 19:     return 0                  # A,8  or better
            if player == 18:     return 0 if dealer in (2,7,8) else 1
            return 1                                      # A,7  or worse
        else:                           # ----- hard hand -----
            if player >= 17:     return 0
            if player >= 13:     return 0 if 2 <= dealer <= 6 else 1
            if player == 12:     return 0 if 4 <= dealer <= 6 else 1
            if player <= 11:     return 1
        return 1                        # default safety

    # --------------------------------------------------------------------- #
    # Bijective state  ⟺  index mapping (360 states)
    # --------------------------------------------------------------------- #
    def s2i(self, state):
        if self.envname == 'blackjack':
            player, dealer, soft = state
            return ((player - 4) * 20) + ((dealer - 1) * 2) + int(soft)
        else:
            return state

    def i2s(self, idx: int) -> Tuple[int, int, bool]:
        soft      = bool(idx & 1)               # faster than idx % 2
        tmp       = idx >> 1                    # integer divide by 2
        dealer    = (tmp % 10) + 1
        player    = (tmp // 10) + 4
        return player, dealer, soft