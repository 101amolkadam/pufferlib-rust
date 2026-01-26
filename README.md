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
- [🏗️ Why Rust for RL?](#️-why-rust-for-rl)
- [🏗️ Architecture](#️-architecture)
- [🚀 Getting Started](#-getting-started)
- [🛠️ Detailed Usage](#️-detailed-usage)
- [📊 Performance & Benchmarking](#-performance--benchmarking)
- [🗺️ Roadmap](#️-roadmap)
- [📜 Technical Specification](SPECIFICATION.md)
- [📜 License](#-license)

---

## 🦀 Why Rust for RL?

While Python is the standard for RL research, it suffers from several bottlenecks that Rust solves natively:

- **True Parallelism**: Rust's thread-safe model allows us to step thousands of environments simultaneously without the memory overhead of Python's `multiprocessing` or the limitations of the Global Interpreter Lock (GIL).
- **Embedded Deployment**: PufferLib Rust can be compiled into a single static binary, making it ideal for RL on edge devices, robots, or in-game AI without a heavy Python runtime.
- **Memory Safety**: Our Emulation Layer uses Rust's ownership system to ensure zero-copy data flow from simulation to neural network, eliminating the large-scale "pickling" overhead common in process-based vectorization.

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
| **Emulation Layer** | Native handling of complex/nested observation spaces. | ✅ |

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
Requires **LibTorch** and **Protocol Buffers**.

**LibTorch Setup (Windows/Linux/Mac):**
```powershell
# Windows
.\setup_libtorch.ps1
```

**Protocol Buffers (Required for RPC/gRPC):**
- **Windows**: `winget install -e --id ProtocolBuffers.Protoc`
- **Ubuntu**: `sudo apt install protobuf-compiler`
- **MacOS**: `brew install protobuf`

### 3. Quick Run
Train a baseline agent on CartPole:
```bash
cargo run --release --bin puffer -- train cartpole --timesteps 1000000
```

    
> [!TIP]
> **New in v0.1.0**: The CLI now includes a real-time throughput dashboard with a progress bar and live SPS (Steps Per Second) metrics.

## 🛠️ CLI Usage Workflows

**1. Basic Training**
```bash
cargo run --release --bin puffer -- train cartpole --timesteps 1000000
```

**2. Resume from Checkpoint**
Seamlessly continue training from a saved state:
```bash
cargo run --release --bin puffer -- train cartpole --resume data/checkpoint_000100.pt
```

**3. Dynamic Difficulty (Curriculum)**
Enable automated difficulty adjustment based on agent performance (ELO):
```bash
# Scales environment entropy/difficulty as agent ELO increases
cargo run --release --bin puffer -- train cartpole --curriculum simple
```

**4. Evaluate Agent**
Run inference without training to verify behavior:
```bash
cargo run --release --bin puffer -- eval cartpole --episodes 10
```

**5. List Environments**
See all supported environments and their properties:
```bash
cargo run --release --bin puffer -- list
```

---

### Performance & Benchmarking

PufferLib Rust is built for extreme throughput. You can verify the performance on your hardware using the built-in benchmark tool:

```bash
# Run a standardized throughput test (CartPole)
cargo run --release --bin puffer -- bench --env cartpole --num-envs 16
```

_Note: Windows users may need to adjust stack size for high `num-envs` counts._

| Environment | Python (Steps/s) | Rust (Steps/s) | Speedup |
| :--- | :---: | :---: | :---: |
| CartPole | ~5k | **~500k+** | **100x** |
| Bandit | ~10k | **~1M+** | **100x** |

> [!NOTE]
> We are currently preparing standardized benchmarks. If you're interested in contributing to our profiling suite, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 🗺️ Roadmap

We have a phased plan to reach v1.0.0, focusing on feature parity with the Python core followed by unique Rust-first optimizations.

See the full [ROADMAP.md](ROADMAP.md) for details.

---

## 🤝 Community & Support

- **Discord**: Join the [PufferAI Discord](https://discord.gg/puffer) for research discussions.
- **Original Project**: [PufferAI/PufferLib](https://github.com/PufferAI/PufferLib) (Python/C).
- **Issues**: Use GitHub Issues for bugs and RFCs (Request for Comments).

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Credits

Inspired by the original [PufferLib](https://puffer.ai) project by JOSEPH SUAREZ. This port aims to provide a high-fidelity Rust implementation of the PufferLib architecture.
