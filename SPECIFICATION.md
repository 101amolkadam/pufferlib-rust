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

## 3. Algorithmic Nuances
### 3.1 V-trace Implementation
- Must support off-policy corrections for asynchronous vectorization.
- Coefficients ($\rho$, $c$) must be clamped and computed in 32-bit floats.

### 3.2 Priority Replay
- Transition to a segment-tree based approach for logarithmic sampling performance.
- Direct integration with the `ExperienceBuffer`.
