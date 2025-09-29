"""Create grid world environments from string layouts
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import os
import random
from tqdm import tqdm


def solve_mdp(P, R, gamma, policies):
    """ Batch Policy Evaluation Solver

    We denote by 'A' the number of actions, 'S' for the number of
    states, 'N' for the number of policies evaluated and 'K' for the
    number of reward functions to evaluate.

    Args:
      P (numpy.ndarray): Transition function as (A x S x S) tensor
      R (numpy.ndarray): Reward function as a (S x A x K) tensor
      gamma (float): Scalar discount factor
      policies (numpy.ndarray): tensor of shape (N x S x A)

    Returns:
      tuple (vfs, qfs) where the first element is a tensor of shape
      (N x S X K) and the second element contains the Q functions as a
      tensor of shape (N x S x A x K).
    """
    nstates = P.shape[-1]
    ppi = np.einsum('ast,nsa->nst', P, policies)
    rpi = np.einsum('sak,nsa->nsk', R, policies)
    vfs = np.linalg.solve(np.eye(nstates) - gamma*ppi, rpi)
    qfs = R + gamma*np.einsum('ast,ntk->nsak', P, vfs)
    return vfs, qfs


def value_iteration(P, R, gamma, num_iters=10):
    """Value iteration for the Bellman optimality equations

    Args:
        P (np.ndarray): Transition function as (A x S x S) tensor
        R (np.ndarray): Reward function as a (S x A) matrix
        gamma (float): Discount factor
        num_iters (int, optional): Defaults to 10. Number of iterations

    Returns:
        tuple: value function and state-action value function tuple
    """
    nstates, nactions = P.shape[-1], P.shape[0]
    qf = np.zeros((nstates, nactions))
    for _ in range(num_iters):
        qf = R + gamma*np.einsum('ast,t->sa', P, np.max(qf, axis=1))
    return np.max(qf, axis=1), qf


def layout_to_array(layout, wall='w'):
    """Convert string layout to array representation

    Args:
        layout (str): Multi-line string where each line is a row of the grid world
        wall (str, optional): Defaults to 'w'. Character describing a wall

    Returns:
        np.ndarray: Numpy array (np.float) where a wall is represented by 0 and 1 for an empty cell.
    """

    return np.array([list(map(lambda c: 0 if c == 'w' else 1, line))
                     for line in layout.splitlines()])


def make_adjacency(layout, epsilon=0.15):
    """Convert a grid world layout to an adjacency matrix with stochastic transitions.

    For each state and chosen action:
      - With probability (1 - epsilon): take the intended move.
      - With probability epsilon: take one of the alternative actions uniformly.

    Args:
        layout (str): Multi-line string layout of the grid.
        epsilon (float): Probability of taking an alternative move.

    Returns:
        tuple: (P, state_to_grid_cell, grid_cell_to_state)
          where P is an array of shape (A, S, S) representing the stochastic transition probabilities.
    """
    # Define directions: UP, DOWN, LEFT, RIGHT.
    directions = [np.array((-1, 0)),  # UP
                  np.array((1, 0)),   # DOWN
                  np.array((0, -1)),  # LEFT
                  np.array((0, 1))]   # RIGHT

    grid = layout_to_array(layout)
    state_to_grid_cell = np.argwhere(grid)
    grid_cell_to_state = {tuple(state_to_grid_cell[s].tolist()): s 
                          for s in range(state_to_grid_cell.shape[0])}
    nstates = state_to_grid_cell.shape[0]
    nactions = len(directions)
    P = np.zeros((nactions, nstates, nstates))

    for state, idx in enumerate(state_to_grid_cell):
        for action, d in enumerate(directions):
            # Determine intended destination.
            intended_idx = idx + d
            if grid[tuple(intended_idx)]:
                intended_dest = grid_cell_to_state[tuple(intended_idx)]
            else:
                intended_dest = state  # hit a wall; remain in same state

            # With probability 1 - epsilon, take the intended action.
            P[action, state, intended_dest] += 1 - epsilon

            # With probability epsilon, randomly choose one of the other directions.
            alternatives = [a for a in range(nactions) if a != action]
            for alt_action in alternatives:
                alt_d = directions[alt_action]
                alt_idx = idx + alt_d
                if grid[tuple(alt_idx)]:
                    alt_dest = grid_cell_to_state[tuple(alt_idx)]
                else:
                    alt_dest = state
                P[action, state, alt_dest] += epsilon / len(alternatives)

    return P, state_to_grid_cell, grid_cell_to_state


# def solve_mdp(P, R, gamma, policies):
#     """ Batch Policy Evaluation Solver

#     We denote by 'A' the number of actions, 'S' for the number of
#     states, 'N' for the number of policies evaluated and 'K' for the
#     number of reward functions to evaluate.

#     Args:
#       P (numpy.ndarray): Transition function as (A x S x S) tensor
#       R (numpy.ndarray): Reward function as a (S x A x K) tensor
#       gamma (float): Scalar discount factor
#       policies (numpy.ndarray): tensor of shape (N x S x A)

#     Returns:
#       tuple (vfs, qfs) where the first element is a tensor of shape
#       (N x S X K) and the second element contains the Q functions as a
#       tensor of shape (N x S x A x K).
#     """
#     nstates = P.shape[-1]
#     ppi = np.einsum('ast,nsa->nst', P, policies)
#     rpi = np.einsum('sak,nsa->nsk', R, policies)
#     vfs = np.linalg.solve(np.eye(nstates) - gamma*ppi, rpi)
#     qfs = R + gamma*np.einsum('ast,ntk->nsak', P, vfs)
#     return vfs, qfs


# def value_iteration(P, R, gamma, num_iters=10):
#     """Value iteration for the Bellman optimality equations

#     Args:
#         P (np.ndarray): Transition function as (A x S x S) tensor
#         R (np.ndarray): Reward function as a (S x A) matrix
#         gamma (float): Discount factor
#         num_iters (int, optional): Defaults to 10. Number of iterations

#     Returns:
#         tuple: value function and state-action value function tuple
#     """
#     nstates, nactions = P.shape[-1], P.shape[0]
#     qf = np.zeros((nstates, nactions))
#     for _ in range(num_iters):
#         qf = R + gamma*np.einsum('ast,t->sa', P, np.max(qf, axis=1))
#     return np.max(qf, axis=1), qf


# def layout_to_array(layout, wall='w'):
#     """Convert string layout to array representation

#     Args:
#         layout (str): Multi-line string where each line is a row of the grid world
#         wall (str, optional): Defaults to 'w'. Character describing a wall

#     Returns:
#         np.ndarray: Numpy array (np.float) where a wall is represented by 0 and 1 for an empty cell.
#     """

#     return np.array([list(map(lambda c: 0 if c == 'w' else 1, line))
#                      for line in layout.splitlines()])


# def make_adjacency(layout, epsilon=0.):
#     """Convert a grid world layout to an adjacency matrix.

#     Args:
#         layout (np.ndarray): Grid layout as an array where 0 means a wall and 1 is empty.

#     Returns:
#         tuple: First element is aulti-dimensional np.ndarray of size (A X S X S) where A=4 is the 
#         number of actions, and S is the number of states. The action set is: 
#         UP (0), DOWN (1), LEFT (2), RIGHT (3). The second element of the tuple is a np.ndarray
#         mapping state (integer) to cell coordinates in the original layout.
#     """
#     directions = [np.array((-1, 0)),  # UP
#                   np.array((1, 0)),  # DOWN
#                   np.array((0, -1)),  # LEFT
#                   np.array((0, 1))]  # RIGHT

#     grid = layout_to_array(layout)
#     state_to_grid_cell = np.argwhere(grid)
#     grid_cell_to_state = {tuple(state_to_grid_cell[s].tolist()): s for s in range(state_to_grid_cell.shape[0])}

#     nstates = state_to_grid_cell.shape[0]
#     nactions = len(directions)
#     P = np.zeros((nactions, nstates, nstates))
#     for state, idx in enumerate(state_to_grid_cell):
#         for action, d in enumerate(directions):
#             # print(tuple(idx + d), grid)
#             if grid[tuple(idx + d)]:
#                 dest_state = grid_cell_to_state[tuple(idx + d)]
#                 P[action, state, dest_state] = 1.
#             else:
#                 P[action, state, state] = 1.

#     return P, state_to_grid_cell, grid_cell_to_state


def build_action_reward(R, P, reward, reward_list):
    """
    Given deterministic transition matrix P (A x S x S) and state reward vector,
    compute R matrix of shape (S x A)
    """
    A, S, _ = P.shape
    for a in range(A):
        for s in range(S):
            next_state = np.argmax(P[a, s])  # only one entry is 1
            if next_state in reward_list:
                R[s, a] = reward
    return R


class TwoRooms:
    """Deterministic four-rooms layout with sparse reward upon reaching the goal
       r: 0 -> sparse reward
          1 -> dense reward
    """

    def __init__(self, reward="sparse-trap"):
        self.layout ="""wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
wwwwwwwwwwwww
"""

        reward_type = reward.split('-')[0]
        target_type = reward.split('-')[1]
        self.reward_type = reward

        self.name = "tworooms"
        self.P, self.state_to_grid_cell, self.grid_cell_to_state = make_adjacency(self.layout, epsilon=0.)  # P [a, s, s']
        self.grid = layout_to_array(self.layout)

        # self.R = np.copy(np.swapaxes(self.P[:, :, -1], 0, 1))
        self.r_dense = 1 if reward_type=="dense" else 0

        if self.r_dense:
            self.dense_resards = np.zeros(len(self.state_to_grid_cell))
            for idx, cell in enumerate(self.state_to_grid_cell):
                self.dense_resards[idx] = - self.compute_manhattan_distances(tuple(cell))

        # breakpoint()
        # self.P[:, -1, :] = 0.
        # self.P[:, -1, -1] = 1.

        if target_type == "trap":
            self.trap_state = [13, 23, 34, 16, 27, 37]
            self.trap_reward = -100.
            self.goal_state = [50]
            self.goal_reward = 1.
            self.end_state = self.trap_state + self.goal_state

            self.P[:, self.end_state, :] = 0.
            A, S, _ = self.P.shape
            R = np.zeros((S, A))
            R = build_action_reward(R, self.P, self.trap_reward, self.trap_state)
            R = build_action_reward(R, self.P, self.goal_reward, self.goal_state)
            
        elif target_type == "sweettrap":
            self.trap_state = [5, 9, 17, 28, 38, 46]
            self.trap_reward = 0.5
            self.goal_state = [50]
            self.goal_reward = 1.
            self.end_state = self.trap_state + self.goal_state

            self.P[:, self.end_state, :] = 0.
            A, S, _ = self.P.shape
            R = np.zeros((S, A))
            R = build_action_reward(R, self.P, self.trap_reward, self.trap_state)
            R = build_action_reward(R, self.P, self.goal_reward, self.goal_state)

        else:
            self.trap_state = []
            self.goal_state = [50]
            self.goal_reward = 1.
            self.end_state = self.goal_state

            self.P[:, self.end_state, :] = 0.
            A, S, _ = self.P.shape
            R = np.zeros((S, A))
            R = build_action_reward(R, self.P, self.goal_reward, self.goal_state)

        self.R = R

        self.discount = 0.99
        self.mdp = [self.P, self.R, self.discount]
        _, self.qf = value_iteration(*self.mdp, num_iters=50)

        self.directions = [np.array((-1, 0)),  # UP
                  np.array((1, 0)),  # DOWN
                  np.array((0, -1)),  # LEFT
                  np.array((0, 1))]  # RIGHT
        
        self.reset()

        # self.MAX_STEPS = 50
        self.MAX_STEPS = 20
        self.avaliable_states = 51 - len(self.end_state)

    def num_states(self, ):
        return len(self.state_to_grid_cell)
    
    def num_actions(self, ):
        return len(self.directions)
    
    def create_dataset(self, data_collecting, state, eps=None):
        if eps is not None:
            action = select_action(self.qf[state, :], eps)
            
        else:
            if data_collecting == 'good':
                action = select_action(self.qf[state, :], 0.5)
            elif data_collecting == 'mid':
                action = np.random.randint(self.num_actions())
            elif data_collecting == 'bad':
                action = select_action(self.qf[state, :], 0.1)
            else:
                raise NotImplementedError(f"data_collecting is {data_collecting}, it should be chosen from good, mid, bad")
        
        # return select_action(self.qf[state, :], 1.)
        return action

    def reset(self, ):
        while True:
            init_cell = [np.random.randint(1, 6), np.random.randint(1, 6)]
            self.state = self.grid_cell_to_state[tuple(init_cell)]
            if self.state in self.end_state:
                continue
            else:
                break
        return self.state, None
    
    def step(self, action):
        
        next_state = np.nonzero(self.P[action][self.state])[0][0]
        # next_state = np.random.choice(self.num_states(), p=self.P[action][self.state])

        if self.r_dense:
            reward = self.dense_resards[next_state]
        else:
            reward = 0

        if next_state in self.trap_state:
            reward = self.trap_reward
            done = True
        elif next_state in self.goal_state:
            reward = self.goal_reward
            done = True
        else:
            done = False

        self.state = next_state
        
        return self.state, reward, done, None
    
    def compute_manhattan_distances(self, source, target=(5, 11)):
        """
        Computes the shortest Manhattan distance from a given source to a target cell,
        ensuring movement is restricted to '1' cells only.

        Args:
            source (tuple): (row, col) starting position.
            target (tuple): (row, col) ending position.

        Returns:
            int: The Manhattan distance from source to target (or -1 if unreachable).
        """
        rows, cols = self.grid.shape
        distances = np.full((rows, cols), -1)  # Initialize all distances as -1 (unreachable)
        
        # BFS queue initialized with the source
        queue = deque([source])
        
        # try:
        #     distances[source] = 0
        # except:
        #     print(source, rows, cols)
        #     breakpoint()

        distances[source] = 0  # Distance to itself is 0

        # Possible movements: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            r, c = queue.popleft()

            # If we reached the target, return the distance
            if (r, c) == target:
                return distances[r, c]

            for dr, dc in directions:
                nr, nc = r + dr, c + dc  # New position

                # Check if inside bounds, is a '1' cell, and has not been visited
                if 0 <= nr < rows and 0 <= nc < cols and self.grid[nr, nc] == 1 and distances[nr, nc] == -1:
                    distances[nr, nc] = distances[r, c] + 1  # Update distance
                    queue.append((nr, nc))  # Push to queue for further expansion

        return -1  # If no valid path exists



def plot_policy(ax, layout, mdp):
    """Solve the MDP and plot the greedy policy on the original grid layout.

    Args:
        ax (matplotlib.pyplot.axis): Axis object for the given figure
        layout (str): Multi-line string where each line is a row of the grid world
        mdp (tuple): (P, R, gamma) where P is (A x S x S), R is (S x A) and gamma is a float [0,1)
    """

    grid = layout_to_array(layout)
    _, state_to_grid_cell, _ = make_adjacency(layout, epsilon=0.)
    ax.imshow(grid)

    action_symbols = ['↑', '↓', '←', '→']
    _, qf = value_iteration(*mdp, num_iters=50)
    for s, c in enumerate(state_to_grid_cell):
        symb = action_symbols[np.argmax(qf[s, :])]
        plt.text(c[1], c[0], symb, color='red', ha='center', va='center')
        ax.text(c[1], c[0], str(s), color='blue', ha='center', va='center')

    gc = state_to_grid_cell[-1]
    ax.text(gc[1], gc[0], 'g')

def select_action(values, p):
    """
    Select an action with probability p for the optimal action (highest value),
    and (1-p) probability for a non-optimal action.

    Args:
        values (numpy array): Array of action values, shape (action_dim,)
        p (float): Probability of selecting the optimal action.

    Returns:
        int: Selected action index.
    """
    # Find the maximum value
    max_value = np.max(values)

    # Identify indices of the optimal actions (multiple if there are ties)
    optimal_actions = np.where(values == max_value)[0]

    # Identify indices of non-optimal actions
    non_optimal_actions = np.where(values != max_value)[0]

    if np.random.rand() < p:
        # Choose one of the optimal actions randomly
        return np.random.choice(optimal_actions)
    else:
        # Choose one of the non-optimal actions uniformly if available
        if len(non_optimal_actions) > 0:
            return np.random.choice(non_optimal_actions)
        else:
            return np.random.choice(optimal_actions)  # Only optimal actions exist



if __name__ == "__main__":
    env = TwoRooms()

    fig, ax = plt.subplots()
    plot_policy(ax, env.layout, env.mdp)

    # plt.show()
    plt.savefig("tworooms.png")
    plt.close()

    # make_buffer(env, 1., 0)

    # for p in [0.5, 0.25, 0.1]:
    #     for seed in [0, 1, 2, 3, 4]:
    #         make_buffer(env, p, seed)

