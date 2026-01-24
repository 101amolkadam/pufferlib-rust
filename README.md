# 🐡 PufferLib Rust

[![Rust CI](https://github.com/101amolkadam/pufferlib-rust/actions/workflows/rust.yml/badge.svg)](https://github.com/101amolkadam/pufferlib-rust/actions/workflows/rust.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-blue.svg)](https://www.rust-lang.org)

**High-performance reinforcement learning library implemented in pure Rust.**

PufferLib Rust is a lightweight, high-performance port of [PufferLib](https://github.com/pufferai/pufferlib). It brings modern reinforcement learning to the Rust ecosystem, eliminating Python overhead while maintaining a clean and expressive API.

---

## 🔥 Features

- 🏎️ **Native Performance**: Built from the ground up for speed with parallel environment execution.
- 🦀 **Pure Rust**: No Python dependencies. Just standard Rust tooling.
- 🧠 **Neural Network Support**: MLP and LSTM policies via `tch-rs` (LibTorch).
- 🛠️ **Extensible API**: Simple `PufferEnv` trait for creating custom environments.
- 📊 **Rich CLI**: Built-in tools for training, evaluation, and visualization.

---

## 🏗️ Project Structure

```text
└── crates/
    ├── pufferlib/           # 🧩 Core engine: spaces, vectorization, PPO training
    ├── pufferlib-envs/      # 🌍 Built-in environments (CartPole, Bandit, etc.)
    └── pufferlib-cli/       # 💻 CLI for training and evaluation
```

---

## 🚀 Getting Started

### 1. Prerequisites

- **Rust**: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **LibTorch** (For Neural Networks):
  ```powershell
  # Windows (PowerShell) - Run the included setup script
  .\setup_libtorch.ps1
  ```

### 2. Quick Training

```bash
# Train on CartPole using the CLI
cargo run --release --bin puffer -- train cartpole --timesteps 100000
```

### 3. Usage as a Library

```rust
use pufferlib::prelude::*;
use pufferlib_envs::CartPole;

fn main() {
    let mut env = CartPole::new();
    let (obs, _info) = env.reset(Some(42));
    
    let action = ArrayD::from_elem(IxDyn(&[1]), 1.0);
    let result = env.step(&action);
    
    println!("Reward: {} | Done: {}", result.reward, result.done());
}
```

---

## 🛠️ Implementation Details

| Module | Description |
| :--- | :--- |
| **Spaces** | Discrete, MultiDiscrete, Box, and Dict support. |
| **Vectorization** | Parallel (Rayon) and Serial execution wrappers. |
| **Training** | Clean PPO implementation with V-trace advantage estimation. |
| **Policies** | Configurable MLP and LSTM architectures. |

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Credits

Based on the original [PufferLib](https://puffer.ai) by Joseph Suarez.
