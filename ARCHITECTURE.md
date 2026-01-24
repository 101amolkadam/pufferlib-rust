# 🏗️ Architecture: PufferLib Rust

This document provides a deep dive into the internal design and implementation of PufferLib Rust.

## 📐 System Overview

PufferLib Rust is designed to be a high-performance, modular framework for reinforcement learning. It follows a "Trait-first" philosophy, allowing developers to swap out environments, vectorization strategies, and training algorithms with minimal friction.

### System Components

```mermaid
graph TD
    subgraph Emulation[Emulation Layer]
        A[PufferEnv Trait] --> S[Space Flattening]
        S --> P[Agent Padding]
    end

    subgraph Vector[Vectorization Layer]
        V[VecEnv Backend] --> SER[Serial]
        V --> PAR[Parallel Rayon]
    end

    subgraph Train[Training Engine]
        B[Exp Buffer] --> PPO[PPO + V-trace]
        PPO --> OPT[Optimizer]
    end

    subgraph Network[Policy Network]
        L[LibTorch] --> MLP[MlpPolicy]
        L --> Rec[LSTMPolicy]
    end

    Emulation --> Vector
    Vector  --> Train
    Train   --> Network
    Network --> Vector
```

## 🧩 Key Modules

### 1. `pufferlib::spaces`
Implements the core observation and action spaces. Supports complexity through recursion (e.g., `Dict` spaces containing `Box` and `Discrete` spaces).
- **Optimization**: Uses `ndarray` for efficient multi-dimensional array manipulation.

### 2. `pufferlib::vector`
Handles the execution of multiple environment instances.
- **Serial**: Runs environments in a single thread, ideal for debugging.
- **Parallel**: Leverages `Rayon` for work-stealing parallelism, maximizing CPU utilization during rollout collection.

### 3. `pufferlib::policy`
Bindings to `libtorch` via the `tch` crate.
- **MLP**: standard multi-layer perceptron for simple state-based environments.
- **LSTM**: Recurrent architecture for POMDPs (Partially Observable Markov Decision Processes).

### 4. `pufferlib::training`
The PPO implementation.
- **V-trace**: Implements V-trace for off-policy corrections, keeping training stable even with asynchronous updates.
- **GAE**: Generalized Advantage Estimation for variance reduction.

## 🛡️ The Emulation Layer
The core innovation of PufferLib is the **Emulation Layer**. It acts as a bidirectional translator between:
1. **Dynamic Simulations**: Environments with variable numbers of agents, heterogeneous action spaces, and nested observations.
2. **Static Neural Networks**: Policies that require fixed-size tensor inputs and constant action dimensions.

Rust achieves this through `DynSpace` and recursive flattening, ensuring that any complex simulation can be trained by a standard MLP or LSTM without modifying the agent code.

## 🔌 Hardware Backends
Currently, PufferLib Rust utilizes **LibTorch** (`tch-rs`) for its high-performance tensor engine. Our architecture is designed to eventually support:
- **Candle**: HuggingFace's pure-Rust ML framework for zero-dependency distributed deployment.
- **ArrayFire**: For high-performance GPGPU computing outside the Torch ecosystem.

## ⚡ Performance Considerations

PufferLib Rust achieves superior performance by:
1. **Zero-copy Transfers**: Observations are moved into buffers with minimal copying.
2. **Native Math**: Utilizing BLAS/LAPACK backends through LibTorch and ndarray for high-speed tensor operations.
3. **No GIL**: Unlike Python RL libraries, PufferLib Rust is free from Global Interpreter Lock issues, allowing true multi-threaded environment stepping.

## 🛠️ Data Flow

1. **Reset**: All environments are initialized with seeds.
2. **Rollout**: The `Trainer` coordinates with the `VectorizedEnv` to collect $N$ steps of experience.
3. **Compute**: Advantages and Returns are calculated in the `ExperienceBuffer`.
4. **Update**: The `Policy` network is updated using mini-batch SGD over multiple epochs.
5. **Evaluate**: Statistics are logged, and the loop continues.
