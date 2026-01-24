# 🐡 PufferLib Rust

[![Rust CI](https://github.com/101amolkadam/pufferlib-rust/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/101amolkadam/pufferlib-rust/actions/workflows/rust.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-blue.svg)](https://www.rust-lang.org)
[![Docs](https://img.shields.io/badge/docs-rs-green)](https://docs.rs/pufferlib-rust)

**High-performance, pure-Rust reinforcement learning library optimized for research and production scale.**

PufferLib Rust is the high-bandwidth link between complex simulations and neural policies. While inspired by the original Python [PufferLib](https://github.com/PufferAI/PufferLib), this port is built from the ground up for **zero-GIL parallelism**, **type-safe emulation**, and **embedded performance**.

---

## 📖 Table of Contents
- [🔥 Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🚀 Getting Started](#-getting-started)
- [🛠️ Detailed Usage](#️-detailed-usage)
- [📊 Performance](#-performance)
- [🗺️ Roadmap](#️-roadmap)
- [📜 License](#-license)

---

## 🔥 Features

| Feature | Description | Status |
| :--- | :--- | :---: |
| **Pure Rust** | Zero Python dependencies. Fast, stable, and safe. | ✅ |
| **Rayon Vectorization** | Multi-threaded environment execution with work-stealing. | ✅ |
| **Off-Policy RL** | Robust PPO implementation with V-trace corrections. | ✅ |
| **Neural Backends** | High-speed tensor operations via LibTorch (`tch-rs`). | ✅ |
| **LSTM Policies** | Full support for temporal dependency tracking. | ✅ |
| **Modular Spaces** | Discrete, Box, Dict, and Multi-discrete actions/obs. | ✅ |
| **Emulation Layer** | Native handling of complex/nested observation spaces. | 🏗️ |

### 🦀 Rust vs. 🐍 Python
| Capability | PufferLib (Python) | PufferLib (Rust) |
| :--- | :---: | :---: |
| Parallelism | Process-based (pickling overhead) | Thread-based (Zero-copy) |
| Safety | Dynamic / Runtime checks | Compile-time / Type-safe |
| Latency | High (Inter-process) | Ultra-low (Local memory) |
| Scalability | Ray Distributed | Rayon Work-stealing |
| Portability | Heavy runtime (Python + Torch) | Static binary (LibTorch only) |

---

## 🏗️ Architecture

PufferLib Rust utilizes a modular architecture that decouples environment logic from training dynamics.

```mermaid
graph LR
    subgraph Environments
        E1[CartPole]
        E2[Bandit]
        E3[Custom]
    end
    
    subgraph Vectorization
        V[Parallel Wrapper]
    end
    
    subgraph Training
        B[Exp Buffer]
        T[PPO Trainer]
    end
    
    subgraph Policy
        P[Linear/LSTM]
    end

    Environments --> V
    V --> B
    B --> T
    T --> P
    P --> V
```

> [!TIP]
> For a technical deep-dive, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 🚀 Getting Started

### 1. Installation
Add `pufferlib` to your `Cargo.toml`:
```toml
[dependencies]
pufferlib = { git = "https://github.com/101amolkadam/pufferlib-rust" }
```

### 2. Native Dependencies
Requires **LibTorch**. Use our automated setup script:
```powershell
# Windows
.\setup_libtorch.ps1
```

### 3. Quick Run
Train a baseline agent on CartPole:
```bash
cargo run --release --bin puffer -- train cartpole --timesteps 1000000
```

---

## 🛠️ Detailed Usage

### Custom Environments
Implement the `PufferEnv` trait to bring your own simulations to life:
```rust
impl PufferEnv for MySimulator {
    fn observation_space(&self) -> DynSpace { ... }
    fn step(&mut self, action: &ArrayD<f32>) -> StepResult { ... }
    // Full implementation in crates/pufferlib-envs
}
```

### Off-Policy Correction (V-trace)
PufferLib uses V-trace to handle the discrepancy between the behavior policy and the target policy, essential for high-throughput vectorized training.

---

## 🗺️ Roadmap

We have a phased plan to reach v1.0.0, focusing on feature parity with the Python core followed by unique Rust-first optimizations.

See the full [ROADMAP.md](ROADMAP.md) for details.

---

## 🤝 Community

We welcome contributions! Please see:
- [CONTRIBUTING.md](CONTRIBUTING.md) for developer guidelines.
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community standards.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Credits

Inspired by the original [PufferLib](https://puffer.ai) project. Designed for performance, safety, and scale.
