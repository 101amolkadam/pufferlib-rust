# 🗺️ PufferLib Rust Roadmap

This document outlines the strategic direction and planned features for the PufferLib Rust ecosystem. Our goal is to provide the fastest, most reliable RL framework in existence.

## 🟢 Phase 1: Core & Emulation (Current Focus)
*Goal: Achieve feature parity with PufferLib's core emulation logic.*

- [x] **Base Traits**: Robust `PufferEnv` and `VecEnv` abstractions.
- [x] **Vectorization**: Serial and Parallel (Rayon-based) backends.
- [x] **Basic Spaces**: Discrete, MultiDiscrete, Box, and Dict support.
- [ ] **Advanced Emulation**:
    - [ ] Variable agent population padding/masking.
    - [ ] Heterogeneous agent handling.
    - [ ] Deep nested observation space flattening (recursive Dict/Tuple).
- [ ] **State Serialization**: Deterministic environment state save/restore.

## 🟡 Phase 2: Algorithmic Expansion
*Goal: Broaden the library's utility beyond standard PPO.*

- [ ] **Continuous Actions**: Full support for Gaussian policies and SAC (Soft Actor-Critic).
- [ ] **Advanced PPO Features**:
    - [ ] Dual-clipping (standard in PufferLib).
    - [ ] Adaptive KL penalty tracking.
    - [ ] Value function clipping.
- [ ] **Protein (Auto-Tune)**: Native Rust implementation of PufferLib's Bayesian HPO system.
- [ ] **Self-Play & ELO**: 
    - [ ] Historical self-play wrappers.
    - [ ] Multi-agent rating systems (ELO/Glicko).
    - [ ] Curriculum scaling based on agent skill.

## 🟠 Phase 3: Hardware & Performance
*Goal: Push the boundaries of RL throughput.*

- [ ] **Candle Backend**: Support for HuggingFace `candle` as an alternative to LibTorch for easier deployment.
- [ ] **Zero-Copy Batching**: Implement cross-process shared memory backends for Linux (via `memfd`).
- [ ] **SIMD Optimization**: Accelerated space-flattening operations.
- [ ] **GPU-Native Envs**: Integration with CUDA-accelerated environment simulations.

## 🔴 Phase 4: Integrations & Ecosystem
*Goal: Making PufferLib the standard for Rust RL.*

- [ ] **Bevy Integration**: Official plugin for 3D physics-based RL in the Bevy engine.
- [ ] **PyO3 Bindings**: Expose Rust environments to Python RL frameworks with zero overhead.
- [ ] **WebAssembly**: Run inference and simple training directly in the browser.
- [ ] **Distributed Scales**: Multi-node vectorization using high-performance networking crates.

---

### 📈 Versioning Strategy
- **v0.x**: Rapid prototyping and API evolution.
- **v1.0**: Stable API, full documentation, and production readiness.
