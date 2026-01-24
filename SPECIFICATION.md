# 📄 Technical Specification: PufferLib Rust

This document defines the rigorous technical standards and implementation details required to achieve feature parity with the original PufferLib (v3.0.0+) while leveraging Rust's safety and performance.

## 1. The Emulation Layer (Standardized Interface)
The core mission of the Emulation Layer is to transform non-standard simulations into "Atari-like" fixed-tensor environments.

### 1.1 Space Flattening
- **Requirement**: Recursive flattening of nested `Dict` and `Tuple` spaces.
- **Implementation**:
    - Build a `SpaceTree` metadata representation during environment initialization.
    - Flatten observations into a single-dimensional `ndarray<f32/u8>`.
    - Provide a zero-copy "unflattener" for policy networks that restores structure as needed.

### 1.2 Agent Population Control (Padding)
- **Problem**: Multi-agent environments (MARL) often have variable agent counts per step.
- **Solution**: The Emulation Layer must pad observations and actions to a constant `max_agents` count.
- **Masking**: Integrate an `action_mask` and `observation_mask` (padding mask) directly into the flattened buffers.

---

## 2. High-Performance Vectorization
Rust's parallelism model allows for tighter memory integration than Python's multiprocessing.

### 2.1 Backend Specifications
- **Serial**: 1:1 mapping of single environment steps.
- **Parallel (Thread-local)**: Rayon-based execution where each worker thread maintains its own environment pool.
- **Parallel (Shared-Memory)**: Future implementation for cross-crate communication using zero-copy memory buffers.

### 2.2 Buffer Management
- **Zero-Copy Rollouts**: Observations and rewards must be written directly into pre-allocated `ExperienceBuffer` memory segments.
- **Batching**: Support for "Synchronous" and "Staggered" step execution (simulating PufferLib's staggered EnvPool).

---

## 4. "Ocean" Environment Philosophy
Following PufferLib 3.x, our "Ocean" equivalent (`pufferlib-envs`) must prioritize raw simulation speed.

### 4.1 Native Rust Speed
- **Requirement**: No virtual machine or heavy runtime overhead.
- **FFI**: Support for direct C/C++ environment bindings via `bindgen`, allowing us to pull in existing PufferLib Ocean environments with zero-copy overhead.

### 4.2 Registry-less Design
- **Decision**: PufferLib Rust will **NOT** implement a central environment registry.
- **Reasoning**: Registries add maintenance burden and obscure type safety.
- **Mechanism**: Use simple factory functions or direct instantiation. Compatibility is achieved via the `PufferEnv` trait.

---

### 5.2 Weight Initialization
- **Requirement**: Orthogonal initialization with gains tailored to activation functions (ReLU, Tanh, Gelu).
- **Bias**: Biases must be initialized to 0.0 unless specified for specific architectures (e.g., LSTM forget gate bias).

---

## 6. Staggered Batching (EnvPool-Lite)
To maximize GPU utilization, PufferLib Rust should implement a non-blocking rollout collection.

### 6.1 Asynchronous Collection
- **Mechanism**: Use a pool of `M` environments to fill a batch of `N` (where `M > N`).
- **Wait Policy**: The trainer does not wait for all environments to step. It pulls the first `N` available results, updating the behavior policy asynchronously.

### 6.2 Shared Flag Synchronization
- **Optimization**: Use atomic flags (`std::sync::atomic`) instead of OS-level mutexes for thread synchronization between environment workers and the training master.
