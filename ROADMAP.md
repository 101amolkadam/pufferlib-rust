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
- [x] **Safe RL**: Constrained PPO with Lagrangian multiplier method.
- [x] **Self-Play & ELO**: 
    - [x] Historical self-play wrappers.
    - [x] Multi-agent rating systems (ELO/Glicko).
    - [x] Curriculum scaling based on agent skill.

## 🟡 Phase 3: Hardware & Performance
*Goal: Push the boundaries of RL throughput.*

- [/] **Candle Backend**: Initial MLP support implemented. Full Trainer/Buffer support pending.
- [x] **Zero-Copy Batching**: Windows Shared Memory backend implemented.
- [ ] **SIMD Optimization**: Accelerated space-flattening operations.
- [ ] **GPU-Native Envs**: Future integration path via `ObservationBatch`.

## 🟡 Phase 4: Integrations & Ecosystem
*Goal: Making PufferLib the standard for Rust RL.*

- [x] **Bevy Integration**: Official plugin for 3D physics-based RL.
- [x] **Gymnasium-Rust**: Initial bridge implementation.
- [x] **PyO3 Bindings**: Python integration for PufferEnv.
- [ ] **WebAssembly**: Placeholder implemented. Full browser training pending.
- [x] **Distributed Scale**: gRPC (tonic) client/server architecture.

---

### 📈 Versioning Strategy
- **v0.x**: Rapid prototyping and API evolution.
- **v1.0**: Stable API, full documentation, and production readiness.
