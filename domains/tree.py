import numpy as np
import os
import pickle
import math
import random
from tqdm import tqdm

import pickle


class TreeEnv:
    def __init__(self, depth, reward='incremental'):
        self.name = 'tree'
        self.MAX_STEPS = 1e5
        self.reward_type = reward

        # Number of total states (nodes) in a full binary tree with 5 layers
        self.depth = depth
        self.n_states = 2 ** depth - 1
        
        # The final layer (leaves) starts at state 15 and goes to state 30
        self.leaf_start = 2 ** (depth-1) - 1
        
        # The root node
        self.root_state = 0
        self.end_state = 1e5
        
        # Current state of the agent
        self.current_state = self.root_state

        # Customized reward
        if reward=='incremental':
            self.reward = self.generate_tree_values(depth+1)
        else:
            self.reward = self.generate_random_values(depth+1)

        # print(self.reward)
        self.reset()
        _, self.pi_opt = self.value_iteration()
        self.reset()

    def rm(self, s, a, s_, move):
        # if a == 0:
        #     next_state = 2 * s + 1
        # elif a == 1:
        #     next_state = 2 * s + 2
        next_state = s_
        return self.reward[next_state]

    def find_layer(self, node_number):
        return math.floor(math.log2(node_number + 1)) 

    def num_states(self):
        return self.n_states
    
    def num_actions(self):
        return 2
    
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
    
    def optimal_action(self, state):
        return 0
        
    def reset(self):
        """
        Reset the environment to the root of the tree.
        Returns:
            state (int): The initial state (always 0 for the root).
        """
        self.current_state = self.root_state
        return int(self.current_state), None

    def step(self, action):
        """
        Take a step in the environment given an action.
        
        Args:
            action (int): 0 or 1.
                          - 0 => 85% chance left (+1 reward), 15% right (-1)
                          - 1 => 85% chance right (-1 reward), 15% left (+1)
                          
        Returns:
            next_state (int): The next state after applying the action.
            reward (float): +1 if going left, -1 if going right.
            done (bool): True if the next state is a leaf, False otherwise.
        """
        # done = False
        
        # Sample a random number to decide left vs. right
        prob = np.random.rand()
        move = ''
        
        if action == 0:
            # Action 0: likely go left
            if prob < 0.85:
                next_state = 2 * self.current_state + 1  # left child
                move = 'left'
                # reward = self.rm()
            else:
                next_state = 2 * self.current_state + 2  # right child
                move = 'right'
                # reward = -self.rm()
        else:
            # Action 1: likely go right
            if prob < 0.85:
                next_state = 2 * self.current_state + 2  # right child
                move = 'right'
                # reward = -self.rm()
            else:
                next_state = 2 * self.current_state + 1  # left child
                move = 'left'
                # reward = self.rm()

        reward = self.rm(self.current_state, action, next_state, move)
        # self.current_state = next_state
        
        # Check if we have reached a leaf
        if self.current_state >= self.leaf_start:
            done = True
            self.current_state = self.end_state
        else:
            done = False
            self.current_state = next_state
        
        return int(self.current_state), reward, done, None
    
    def generate_tree_values(self, n):
        """
        Generate a dictionary {node_index: value} for a complete binary tree
        of n layers (2^n - 1 nodes), following the rules:

        - Node 0 has value = 0 (no direction).
        - A left child continues the parent's direction if the parent was left,
            otherwise it resets to +1.
        - A right child continues the parent's direction if the parent was right,
            otherwise it resets to -1.
        - "Continuing direction" means we double the parent's absolute value.
        - "Changing direction" means we reset the magnitude to 1.
        - Sign is positive for left, negative for right.
        """
        
        if n < 1:
            return {}
        
        num_nodes = 2**n - 1
        values = [0] * num_nodes  # Pre-allocate array to store node values
        
        # Root node value is 0 (special case)
        values[0] = 0
        
        # Compute values for subsequent nodes
        for i in range(1, num_nodes):
            parent = (i - 1) // 2
            # Determine if i is left child or right child
            is_left_child = (i == 2 * parent + 1)
            
            # Parent's value
            parent_value = values[parent]
            
            # Determine parent's "direction" from its sign
            # (Only needed if parent != root, but for root the direction is None.)
            if parent == 0 and parent_value == 0:
                parent_direction = None
            else:
                parent_direction = 'L' if parent_value > 0 else 'R'
            
            # Current direction is 'L' or 'R'
            current_direction = 'L' if is_left_child else 'R'
            
            if parent_direction is None:
                # The root has no direction, so we always reset to ±1
                magnitude = 1
            else:
                # If same direction, double the parent's magnitude
                # If different direction, reset to 1
                if current_direction == parent_direction:
                    magnitude = abs(parent_value) * 2
                else:
                    magnitude = 1
            
            # Assign sign based on current direction
            sign = +1 if is_left_child else -1
            
            # Final value
            values[i] = sign * magnitude
        
        # Convert to a dictionary {index: value}
        return {i: values[i] for i in range(num_nodes)}
    
    def generate_random_values(self, n):

        random_reward_path = "RLLF/domains/tree_reward.pkl"
        if os.path.exists(random_reward_path):
            with open(random_reward_path, 'rb') as f:
                return pickle.load(f)

        else:
            num_nodes = 2**n - 1
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
        # nS = 2 ** (self.depth + 1) - 1    
        nA = self.num_actions()
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        for s in range(nS):
            done = s >= self.leaf_start

            ns = 2 * s + 1
            P[s][0].append((0.85, ns, self.reward[ns], done))
            ns = 2 * s + 2
            P[s][0].append((0.15, ns, self.reward[ns], done))
            
            ns = 2 * s + 2
            P[s][1].append((0.85, ns, self.reward[ns], done))
            ns = 2 * s + 1
            P[s][1].append((0.15, ns, self.reward[ns], done))

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
        # nS = 2 ** (self.depth + 1) - 1    
        nA = self.num_actions()
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
                        q_vals[a] += prob * (r + (0 if done else gamma * V[next_s]))
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
                    q_vals[a] += prob * (r + (0 if done else gamma * V[next_s]))
            policy[s] = np.argmax(q_vals)
        return V, policy


