# ⌨️ CLI Specification: puffer

The `puffer` CLI is the primary entry point for training and evaluating agents. It must be consistent, performance-oriented, and CI-friendly.

## 1. Core Commands

### 1.1 `puffer train`
Train a new policy on a specific environment.
- **Arguments**:
    - `<ENV>`: The environment name (e.g., `cartpole`, `bandit`).
- **Flags**:
    - `--timesteps <N>`: Total timesteps to train (default: 100K).
    - `--batch-size <N>`: Rollout batch size (default: 2048).
    - `--lr <F>`: Learning rate.
    - `--use-vtrace`: Enable off-policy correction.
    - `--use-lstm`: Use a recurrent policy architecture.
- **Output**: Real-time TUI dashboard showing SPS (Steps Per Second), loss, and rewards.

### 1.2 `puffer eval`
Evaluate a trained checkpoint or a random policy.
- **Arguments**:
    - `<ENV>`: Environment name.
- **Flags**:
    - `--checkpoint <PATH>`: Path to `.pt` file (optional).
    - `--episodes <N>`: Number of episodes to run (default: 10).
    - `--render`: Enable environment rendering.

### 1.3 `puffer sweep`
Perform hyperparameter optimization (Phase 2).
- **Mode**: Bayesian Optimization (Protein algorithm).

## 2. Configuration Strategy
- **Prioritization**: CLI Flags > Environment Variables > `config.ini` (Phase 2).
- **Format**: Standardization on `toml` or `ini` for persistent configuration files.
