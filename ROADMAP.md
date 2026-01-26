# 🗺️ PufferLib Rust Roadmap

This document outlines the strategic direction and planned features for the PufferLib Rust ecosystem. Our goal is to provide the fastest, most reliable RL framework in existence.

## 🟢 Phase 1: Core & Emulation (COMPLETE)
*Goal: Achieve feature parity with PufferLib's core emulation logic.*

- [x] **Base Traits**: Robust `PufferEnv` and `VecEnv` abstractions.
- [x] **Vectorization**: Serial and Parallel (Lock-Free Rayon) backends.
- [x] **Basic Spaces**: Discrete, MultiDiscrete, Box, and Dict support.
- [x] **Advanced Emulation**:
    - [x] Variable agent population padding/masking.
    - [x] Heterogeneous agent handling.
    - [x] Deep nested observation space flattening (recursive Dict/Tuple).
- [x] **State Serialization**: Deterministic environment state save/restore.

## � Phase 2: Algorithmic Expansion (COMPLETE)
*Goal: Broaden the library's utility beyond standard PPO.*

- [x] **Continuous Actions**: Gaussian policies and SAC foundation (Entropy Regularization).
- [x] **Advanced PPO Features**:
    - [x] Dual-clipping (standard in PufferLib).
    - [x] Adaptive KL penalty tracking.
    - [x] Value function clipping.
- [x] **Protein (Auto-Tune)**: Native Rust implementation of PufferLib's Bayesian HPO system.
- [x] **Self-Play & ELO**: 
    - [x] Historical self-play wrappers.
    - [x] Multi-agent rating systems (ELO/Glicko).
    - [x] Curriculum scaling based on agent skill.

## � Phase 3: Hardware & Performance
*Goal: Push the boundaries of RL throughput.*

- [x] **Candle Backend**: Support for HuggingFace `candle` as an alternative to LibTorch for easier deployment.
- [x] **Zero-Copy Batching**: Implement cross-process shared memory backends for Linux/Windows (via `memfd`/Named Mappings).
- [x] **SIMD Optimization**: Accelerated space-flattening operations (Zero-allocation `flatten_to`).
- [x] **GPU-Native Envs**: Integration with CUDA-accelerated environments via `ObservationBatch`.

## 🔴 Phase 4: Integrations & Ecosystem
*Goal: Making PufferLib the standard for Rust RL.*

- [x] **Bevy Integration**: Official plugin for 3D physics-based RL in the Bevy engine.
- [x] **Gymnasium-Rust**: Direct bridge to traditional Rust-gym environments.
- [x] **PyO3 Bindings**: Expose Rust environments to Python RL frameworks with zero overhead.
- [x] **WebAssembly**: Run inference and simple training directly in the browser (e.g., via `wasm-bindgen`).
- [x] **Distributed Scales**: Multi-node vectorization using `tonic` (gRPC) or `quinn` (QUIC).

---

### 📈 Versioning Strategy
- **v0.x**: Rapid prototyping and API evolution.
- **v1.0**: Stable API, full documentation, and production readiness.
