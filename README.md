# Which Rewards Matter? Reward Selection for Reinforcement Learning from Limited Feedback

Source code for [Which Rewards Matter? Reward Selection for Reinforcement Learning from Limited Feedback](https://arxiv.org/abs/2510.00144)

> The ability of reinforcement learning algorithms to learn effective policies is determined by the rewards available during training. However, for practical problems, obtaining large quantities of reward labels is often infeasible due to computational or financial constraints, particularly when relying on human feedback. When reinforcement learning must proceed with limited feedback---only a fraction of samples get rewards labeled---a fundamental question arises: which samples should be labeled to maximize policy performance? We formalize this problem of reward selection for reinforcement learning from limited feedback (RLLF), introducing a new problem formulation that facilitates the study of strategies for selecting impactful rewards. Two types of selection strategies are investigated: (i) heuristics that rely on reward-free information such as state visitation and partial value functions, and (ii) strategies pre-trained using auxiliary evaluative feedback. We find that critical subsets of rewards are those that (1) guide the agent along optimal trajectories, and (2) support recovery toward near-optimal behavior after deviations. Effective selection methods yield near-optimal policies with significantly fewer reward labels than full supervision, establishing reward selection as a powerful paradigm for scaling reinforcement learning in feedback-limited settings.

## Installation

### Requirements
```bash
pytorch == 2.6.0
gym == 0.26.2
tqdm
h5py
easydict
omegaconf
```

### MinAtar Setup (Optional)
For MinAtar experiments, install the official [MinAtar](https://github.com/kenjyoung/MinAtar) package:
```bash
git clone https://github.com/kenjyoung/MinAtar.git
cd MinAtar
pip install -e .
```

## Quick Start

### Step 1: Generate Dataset

#### Tabular Domains
```bash
python make_dataset.py domain=graph
python make_dataset.py domain=tree
python make_dataset.py domain=tworooms
```

#### MinAtar Domains (Requires Pre-trained Expert Policy)
For MinAtar domains, you must first train an expert policy and place the model file in the correct location:

**Required Directory Structure:**
```
{root}/{game_name}/model/model_data_and_weights
```

**Example for Breakout:**
```
{root}/breakout/model/model_data_and_weights
```

The expert policy should be a PyTorch checkpoint containing `policy_net_state_dict` compatible with the DQN architecture used in `minatar/dqn.py`.

**Generate MinAtar Dataset:**
```bash
# After placing expert policy in correct location
python make_dataset.py domain=minatar/breakout
python make_dataset.py domain=minatar/freeway
python make_dataset.py domain=minatar/seaquest
python make_dataset.py domain=minatar/asterix
```

**Training Expert Policy:**
Refer to the [official MinAtar repository](https://github.com/kenjyoung/MinAtar) for training expert policies. The trained model should be saved with the key `policy_net_state_dict` and placed at the path specified above.

### Step 2: Run Experiments
```bash
# Heuristic selection strategies
python main.py domain=graph domain.exp.algo=guided domain.exp.impute=zero

# Training phase selection strategies
python main.py domain=graph selection=training_phase selection_params.search=greedy domain.exp.impute=zero

```

## Architecture Overview

### Core Components

#### 1. Selection Strategies (`selection/`)
- **Heuristic Selection** (`heuristics_selection.py`): `guided`, `vistation`, and `uniform`
- **Training Phase Selection** (`training_phase_selection.py`): `brute-force`, `sequential-greedy` and `ES`

#### 2. Domains (`domains/`)
**Tabular Domains:**
- **Graph**: A two-row graph structure with 8 nodes per row.
- **Tree**: A complete binary tree where actions correspond to moving left or right.
- **TwoRooms**: Two $5 \times 5$ gridworld rooms connected by a narrow bottleneck state (variants: TwoRooms-Trap).
- **CliffWalk/FrozenLake**: Classic [RL benchmarks](https://gymnasium.farama.org/environments/toy_text/)

**Deep RL Domains:**
- **MinAtar**: Simplified Atari games (Breakout, Freeway, Seaquest, Asterix)

### Separate Domain Implementations

The framework maintains separate implementations for tabular and image-based domains due to fundamental differences in state representation and indexing:

- **Tabular Domains**: Use discrete integer state indexing where states can be directly mapped to Q-table entries (e.g., `Q[state_id, action]`). This enables efficient exact state lookups and reward selection based on state IDs.
- **Image-based Domains**: Convert high-dimensional image states to compact bit representations using `img2byte()`, then use hash-based indexing via `zlib.crc32()` for state identification and matching. States are indexed by their byte hash rather than direct array comparison, enabling efficient state deduplication and reward selection in large datasets.

This separation ensures optimal performance for each domain type while maintaining clean, specialized codebases.

#### 3. Learning Models (`models/`)
- **Q-Table** (`qtable.py`): Tabular Q-learning for tabular states
- **Deep Q-Network** (`dqn.py`): Deep Q-learning for high-dimensional states

## Configuration 

The framework uses Hydra for hierarchical configuration management:

```yaml
# config/config.yaml
defaults:
  - domain: graph                    # Environment selection
  - selection: training_phase        # Selection strategy

selection_params:
  search: greedy                     # Training phase search method
  eval_episodes: 10000              # Evaluation episodes
  gamma: 0.99                       # Discount factor

dataset:
  data_collecting: 'good'           # Expert policy quality
  dataset_size: 100_000            # Number of transitions

general:
  seed: 0                          # Random seed
  budget: null                     # Selection budget (null = full)
  parallel_num: 1                  # Parallel processes
```

#### Domain Configuration Example
```yaml
# config/domain/graph.yaml
domain:
  _target_: domains.graph            # Domain class instantiation
  max_length: 8                      # Domain specific parameters
  reward: 'one'                      # 
  transitions_deterministic: True    # 

exp:
  each_query: 1                      # Reward selection batch size
  algo: guided                       # Heuristic selection algorithm
  impute: none                       # Policy learning method

  # Q-learning parameters
  qiterations: 10                    # Q-learning iterations
  qalpha: 0.05                      # Q-learning rate

  # Guided selection parameters
  decay: 'linear'                    # Acquisition function decay
  fixtime: 0.5                      # Decay schedule timing
  decay_temp: 6.0                   # Temperature parameter
```

### Key Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `domain` | `graph`, `tree`, `tworooms`, `cliffwalk`, `frozenlake`, `minatar/breakout`, `minatar/freeway`, `minatar/seaquest`, `minatar/asterix`| Environment |
| `selection` | `heuristics`, `training_phase` | Selection strategy type |
| `selection_params.search` | `greedy`, `evolutionary` | Training phase selection strategies |
| `domain.exp.algo` | `uniform`, `visitation`, `guided` | Heuristic selection strategies |
| `domain.exp.impute` | `zero`, `none` | Policy learning from partially reward-labeled data |


The `domain.exp.impute` parameter controls policy learning from partial reward labels:

- **`zero` (UDS)**: Implements [UDS](https://arxiv.org/abs/2202.01741) by replacing unknown rewards with zero, then applying standard Q-learning.
- **`none` (Adaptive Q-learning)**: Sets Q-values of unlabeled states to zero without reward imputation.

## Advanced Usage

### Custom Domains
Implement the domain interface:
```python
class CustomDomain:
    def __init__(self):
        self.name = "custom"
        self.reward_type = "sparse"
        self.MAX_STEPS = 1000

    def reset(self):
        # Return initial state
        pass

    def step(self, action):
        # Return next_state, reward, done, info
        pass

    def num_actions(self):
        # Return action space size
        pass

    def create_dataset(self, policy_type, state):
        # Return action for dataset generation
        pass
```

### Custom Selection Strategies
Extend the base selection class:
```python
from selection.base_selection import base_selection

class CustomSelection(base_selection):
    def __init__(self):
        super().__init__()
        self.selection_name = "custom"

    def run(self):
        # Implement selection logic
        for budget in range(self.each_query, self.total_budget):
            next_visit_ids = self.select_states(budget)
            performance, accuracy = self.iqltrain(next_visit_ids)
```

### Cluster Computing
For large-scale training phase selection experiments, the framework supports cluster job submission. To customize for your cluster environment, **`selection/training_phase_selection.py`** need to be replaced with your specific cluster job submission logic.

#### Customization Points:

**1. Job Scheduler Adaptation (`push_single_job()` method):**
```python
# Current: SLURM-based implementation
submit_cmd = ["sbatch", fn]
check_cmd = ["squeue", "-j", str(job_id)]
```

**2. Batch Script Generation:**
```python
# Modify lines 106-127 in training_phase_selection.py
lines.append(f'#SBATCH --job-name=...')     # Change directive format
lines.append(f'#SBATCH --nodes=1')          # Adjust resource requests
lines.append('source conda.sh')             # Update environment setup
lines.append('cd YOUR_ROOT/RLLF')           # Set correct working directory
```

**3. Environment Variables:**
- **`ROOT`**: Set to your cluster's job submission directory (line 93)
- **`YOUR_ROOT`**: Update paths to your project location (lines 117, 165)

**4. Resource Requirements:**
Adjust memory, time, and CPU allocations based on your cluster's available resources and queue limits.

<!-- ## Citation

If you use this code in your research, please cite our paper:

```bibtex

``` -->
