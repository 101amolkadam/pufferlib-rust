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

### 1.2 `puffer safe-train`
Train a policy with safety constraints using Lagrangian optimization.
- **Flags**:
    - Inherits all `puffer train` flags.
    - `--cost-limit <F>`: Maximum allowed cost per episode (default: 0.1).
    - `--lagrangian-lr <F>`: Learning rate for the multiplier (default: 0.01).
- **Output**: Dashboard includes real-time cost tracking and Lagrangian multiplier value.

### 1.3 `puffer eval`
Evaluate a trained checkpoint or a random policy.
- **Arguments**:
    - `<ENV>`: Environment name.
- **Flags**:
    - `--checkpoint <PATH>`: Path to `.pt` file (optional).
    - `--episodes <N>`: Number of episodes to run (default: 10).
    - `--render`: Enable environment rendering.

### 1.4 `puffer autotune`
Perform Bayesian hyperparameter optimization (Protein).
- **Flags**:
    - `--trials <N>`: Number of optimization trials (default: 20).
    - `--steps <N>`: Timesteps per trial.

### 1.5 `puffer bench`
Measure throughput (SPS) on specific hardware.
- **Flags**:
    - `--duration <N>`: Test duration in seconds.
    - `--num-envs <N>`: Hardware parallelism to test.

### 1.6 `puffer list`
List all officially supported environments and their properties.

### 1.7 `puffer demo`
Run an environment interactively with real-time rendering.

## 2. Configuration Strategy
- **Prioritization**: CLI Flags > Environment Variables > `config.ini` (Phase 2).
- **Format**: Standardization on `toml` or `ini` for persistent configuration files.
