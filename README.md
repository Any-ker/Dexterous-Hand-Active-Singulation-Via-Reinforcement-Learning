# ME5418 Dexterous Hand Reinforcement Learning System

## Overview

This repository contains a full Isaac Gym workflow for dexterous manipulation.  
It trains a ShadowHand-style robot to **grasp** a target object while singulating surrounding clutter using **Proximal Policy Optimization (PPO)** with manually implemented neural network forward/backward passes and an in-house Adam optimizer.  
The codebase provides:

- Custom PPO trainer (`algorithms/rl/ppo/`)
- DexGrasp Isaac Gym task (`dex_grasp.py`)
- Plugin-based visualization/video logging
- CLI tooling to launch training/testing with a single `python run_train.py ...` command

## Repository Structure

```
ME5418/
├── run_train.py                # Main entry: loads config, builds env, starts PPO
├── dex_grasp.py                # Isaac Gym task (replaces legacy dex_grasp2.py)
├── algorithms/
│   └── rl/ppo/
│       ├── module.py           # Actor-Critic network (manual forward/backward)
│       ├── ppo.py              # PPO trainer + optimizer + logging
│       └── storage.py          # Rollout buffer with GAE + minibatch sampler
├── utils/
│   ├── config.py               # Arg parsing and YAML loading
│   ├── parse_task.py           # Connects configs to task/env construction
│   ├── process_sarl.py         # Creates PPO instance ("agent factory")
│   └── general_utils.py        # Shared helpers (YAML IO, seeding, etc.)
├── cfg/
│   ├── ppo/config.yaml         # Network + optimizer hyperparameters
│   └── train/dedicated_policy.yaml  # Task-specific overrides
├── dexgrasp/                   # Plugin framework (visualization, video logging)
├── environment.yml             # Conda environment definition
└── README.md                   # You are here
```

### Key Components

| File/Folder | Role |
|-------------|------|
| `dex_grasp.py` | Defines the entire Isaac Gym task: assets, resets, reward, observations. |
| `algorithms/rl/ppo/module.py` | `ActorCritic` network with manual forward/backward/Adam. |
| `algorithms/rl/ppo/ppo.py` | High-level PPO training/testing loop. |
| `algorithms/rl/ppo/storage.py` | Rollout buffer storing transitions and computing GAE. |
| `utils/process_sarl.py` | Builds a PPO instance from configs (agent factory). |
| `utils/parse_task.py` | Creates DexGrasp task + vectorized environment. |
| `run_train.py` | Orchestrates everything: load config, parse task, build agent, launch training/testing. |

## Software & Hardware Requirements

### Hardware
- NVIDIA GPU with CUDA (RTX 3080 or higher recommended)
- ≥16 GB system RAM
- ≥10 GB disk space

### Software (minimum versions unless noted)
- Ubuntu 20.04+ (tested)
- CUDA 11.7 (set `CUDA_HOME` accordingly)
- Python 3.8
- Conda (for `environment.yml`)
- Isaac Gym Preview 4 (bundled in `isaacgym/` folder)
- Major Python libraries (see `environment.yml` for exact versions):
  - `torch==1.13.0+cu117`
  - `numpy`, `trimesh`, `scikit-learn`, `matplotlib`, `tqdm`, etc.

## Environment Setup (unchanged)

> **Important**: Follow exactly to ensure Isaac Gym + CUDA run correctly.

```bash
# Clone or download the project
cd /path/to/ME5418

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate me5418_dexterous_hand

# Set CUDA environment variables (REQUIRED)
export CUDA_HOME=/path/to/cuda-11.7
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
bash install.sh
```

## Usage

All experiments are launched via `run_train.py`.  
Two canonical commands are provided below—one for training from scratch (with curriculum + resume) and one for testing a saved policy.

### Training

```bash
python run_train.py \
  --task DexGrasp \
  --algo ppo \
  --seed 0 \
  --rl_device cuda:0 \
  --num_envs 256 \
  --max_iterations 6000 \
  --config dedicated_policy.yaml \
  --surrounding_obj_num 4 \
  --model_dir logs/dex_grasp/ppo/ppo_seed0_obj2/model_10000.pt
```

Notes:
- `--model_dir` loads a previous checkpoint (if omitted the trainer starts from scratch).
- `--surrounding_obj_num` toggles between pure grasping (`0`) and clutter singulation (`>0`).

### Testing / Evaluation

```bash
python run_train.py \
  --task DexGrasp \
  --algo ppo \
  --seed 0 \
  --rl_device cuda:0 \
  --config dedicated_policy.yaml \
  --surrounding_obj_num 4 \
  --num_envs 1 \
  --test \
  --test_iteration 10 \
  --model_dir logs/dex_grasp/ppo/ppo_seed0_obj4/model_1500.pt

```

- `--test` switches PPO into evaluation mode (no gradient updates).
- `--test_iteration` controls how many rollout episodes to run.

## Frequently Used Scripts

1. **`run_train.py`**  
   Parses command-line flags, loads YAML configs, constructs DexGrasp environments, and triggers PPO training/testing.

2. **`dex_grasp.py`**  
   Isaac Gym task definition (assets, resets, reward shaping, observation assembly, plugin hooks).

3. **`algorithms/rl/ppo/module.py`**  
   `ActorCritic` network with manual forward/backward & optimizer logic.

4. **`algorithms/rl/ppo/ppo.py`**  
   Implements PPO updates, GAE, logging to TensorBoard/W&B (if enabled).

5. **`algorithms/rl/ppo/storage.py`**  
   Rollout storage that organizes experience and yields mini-batches.

6. **`utils/config.py` / `utils/process_sarl.py` / `utils/parse_task.py`**  
   Glue code to parse CLI flags, read YAML, instantiate tasks + PPO objects.

## Results & Logging

During training/testing, the task prints:
- Success and singulation statistics every reset
- Reward breakdowns (if `Debug.reward_log_interval` > 0)
- TensorBoard logs (success rate, rewards, etc.) in `logs/dex_grasp/ppo/…`
- Optional visualization/video files if enabled via `cfg["env"]["visualization"]` or `video_logging`.

## Baseline Comparison

To validate the effectiveness of the proposed method in cluttered environments, we compared it against a strong baseline policy trained exclusively for direct grasping. Related videos are in the folder.

| Policy | Surrounding Objects | Singulation Success Rate | Notes |
| :--- | :---: | :---: | :--- |
| **Baseline** | 0 | **93%** | Robust in isolation, but lacks singulation logic. |
| **Baseline** | 4 | **0% (0/10)** | Fails completely when the target is blocked by clutter. |
| **Proposed Method** | 4 | **50%** | Successfully pushes obstacles away to grasp the target. |

**Key Observation:** While the baseline excels in clean environments, it cannot handle clutter. Our proposed method bridges this gap, achieving a **50% singulation success rate** in complex scenarios where the baseline fails entirely.

### Demonstration Videos

Below are three demonstration videos showcasing the performance comparison between the baseline and proposed methods:

#### Video 1: Baseline Policy (No Clutter)
Successfully grasps the target object in an unobstructed environment.

https://github.com/user-attachments/assets/4ecc378a-03be-43af-a298-a50affa6de18

#### Video 2: Baseline Policy (With Clutter)
Demonstrates failure when the target object is surrounded by obstacles.

https://github.com/user-attachments/assets/c9f489a2-f87f-48c9-9410-ee9c843ebc93

#### Video 3: Proposed Method (With Clutter)
Shows successful singulation and grasping in cluttered scenarios.

https://github.com/user-attachments/assets/99d5fdd9-07e3-47d8-9e8a-9fd26c1c7ac6



## Troubleshooting

- Ensure Isaac Gym binaries match the Python version (3.8).
- Set `CUDA_HOME` to the *exact* 11.7 toolkit used during compilation.
- If you see `gym_38` errors, re-run `bash install.sh` inside the activated conda env.

## License & Citation

This project is released under the MIT License.  
If you use this codebase in academic work, please cite appropriately.
