# PufferLib Rust

A high-performance reinforcement learning library implemented in pure Rust.

## Overview

PufferLib Rust is a port of [PufferLib](https://github.com/pufferai/pufferlib), bringing high-performance RL training to the Rust ecosystem. It eliminates the need for Python while maintaining the core design principles of simplicity and speed.

## Features

- **Pure Rust** - No Python dependencies, minimal external libraries
- **High Performance** - Native Rust speed with parallel environment execution
- **Clean API** - Simple `PufferEnv` trait for custom environments
- **PPO Training** - Complete PPO implementation with V-trace support
- **Neural Networks** - MLP and LSTM policies via `tch-rs` (libtorch bindings)

## Project Structure

```
└── crates/
    ├── pufferlib/           # Core library
    │   ├── spaces/          # Observation/action spaces
    │   ├── env/             # Environment trait and wrappers
    │   ├── vector/          # Vectorized environments
    │   ├── policy/          # Neural network policies
    │   └── training/        # PPO training system
    │
    ├── pufferlib-envs/      # Built-in environments
    │   ├── bandit.rs        # Multi-armed bandit
    │   ├── cartpole.rs      # CartPole control
    │   ├── squared.rs       # Grid navigation
    │   └── memory.rs        # Sequence memory
    │
    └── pufferlib-cli/       # Command-line interface

```

## Installation

### Prerequisites

1. **Rust** (1.70+): https://rustup.rs/
2. **libtorch** (for neural networks):
   ```bash
   # Download from https://pytorch.org/get-started/locally/
   # Set environment variables:
   export LIBTORCH=/path/to/libtorch
   export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
   ```

### Building

```bash
cd pufferlib-rust
cargo build --release
```

## Usage

### CLI

```bash
# Train on CartPole
cargo run --release --bin puffer -- train cartpole --timesteps 100000

# Train on Bandit
cargo run --release --bin puffer -- train bandit --timesteps 10000

# Evaluate
cargo run --release --bin puffer -- eval cartpole --episodes 10

# List environments
cargo run --release --bin puffer -- list
```

### Library

```rust
use pufferlib::prelude::*;
use pufferlib_envs::CartPole;

fn main() {
    // Create environment
    let mut env = CartPole::new();
    
    // Reset
    let (obs, _info) = env.reset(Some(42));
    
    // Step
    let action = ArrayD::from_elem(IxDyn(&[1]), 1.0);
    let result = env.step(&action);
    
    println!("Reward: {}", result.reward);
    println!("Done: {}", result.done());
}
```

### Custom Environment

```rust
use pufferlib::env::{PufferEnv, EnvInfo, StepResult};
use pufferlib::spaces::{DynSpace, Discrete, Box as BoxSpace};
use ndarray::{ArrayD, IxDyn};

struct MyEnv {
    state: f32,
}

impl PufferEnv for MyEnv {
    fn observation_space(&self) -> DynSpace {
        DynSpace::Box(BoxSpace::uniform(&[1], 0.0, 1.0))
    }
    
    fn action_space(&self) -> DynSpace {
        DynSpace::Discrete(Discrete::new(2))
    }
    
    fn reset(&mut self, _seed: Option<u64>) -> (ArrayD<f32>, EnvInfo) {
        self.state = 0.0;
        (ArrayD::from_elem(IxDyn(&[1]), self.state), EnvInfo::new())
    }
    
    fn step(&mut self, action: &ArrayD<f32>) -> StepResult {
        let a = action.iter().next().unwrap().round() as usize;
        self.state = if a == 1 { 1.0 } else { 0.0 };
        
        StepResult {
            observation: ArrayD::from_elem(IxDyn(&[1]), self.state),
            reward: self.state,
            terminated: true,
            truncated: false,
            info: EnvInfo::new(),
        }
    }
}
```

## Modules

### Spaces

- `Discrete` - Single integer actions
- `MultiDiscrete` - Multiple integer actions
- `Box` - Continuous bounded values
- `Dict` - Dictionary of spaces

### Environments

- `PufferEnv` trait - Core environment interface
- `EpisodeStats` wrapper - Track episode return/length
- `ClipAction` wrapper - Clip continuous actions

### Vectorization

- `Serial` - Sequential execution (debugging)
- `Parallel` - Parallel execution with rayon

### Policies

- `MlpPolicy` - Multi-layer perceptron
- `LstmPolicy` - LSTM for temporal dependencies

### Training

- `Trainer` - PPO training loop
- `ExperienceBuffer` - Rollout storage
- `compute_gae` / `compute_vtrace` - Advantage computation

## License

MIT License - See LICENSE file.

## Credits

Based on [PufferLib](https://puffer.ai) by Joseph Suarez.
