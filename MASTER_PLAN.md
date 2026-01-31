# 🏆 PufferLib Rust: Master Engineering Plan

This plan consolidates the strategic vision from `ROADMAP.md`, technical optimizations from `engineering_plan.md`, and specifications from `SPECIFICATION.md` into a single, high-velocity critical path.

## 🏁 The Engineering Core Philosophy
1. **Performance first**: No abstraction overhead in the hot Path (Monomorphization).
2. **Zero-copy always**: Observations flow directly from env to buffer.
3. **High Fidelity**: Bit-for-bit parity with original PufferLib logic where possible.

---

## 🛤️ Milestone 1: Performance Baseline & Refactor
*Objective: Solidify the foundation before expanding features.*

- [x] **Monomorphization**: Refactor `Serial<E>` and `Parallel<E>` to eliminate `Box<dyn PufferEnv>` and vtable overhead.
- [x] **Zero-Allocation Info**: Replace `HashMap` in `EnvInfo` with static fields and `SmallVec`.
- [x] **Throughput Dashboard**: enhance CLI with real-time SPS (Steps Per Second) and latency metrics.

## Track 2: The Advanced Emulation Layer (DONE)
*Objective: Full compatibility with complex simulations.*

- [x] **Recursive Flattening**: Implement `SpaceTree` for deep `Dict/Tuple` support.
- [x] **Agent Population**: Implement `max_agents` padding and masking for MARL parity.
- [x] **State Serialization**: Deterministic environment state save/restore.
- [x] **Zero-Copy Unflattener**: High-speed structure restoration for policy networks.

## 🛤️ Milestone 3: Algorithmic Expansion (DONE)
*Objective: Broaden utility beyond standard Discrete PPO.*

- [x] **Continuous Actions**: Full support for Gaussian policies and SAC.
- [x] **Advanced PPO Features**: Dual-clipping, adaptive KL, value clipping.
- [x] **Self-Play Wrapper**: Native support for ELO-based training.
- [x] **Safe RL**: Constrained PPO with Lagrangian dual optimization.
- [x] **Multi-Agent**: MAPPO and team-based role support.
- [x] **Sequence-Based**: Decision Transformer and Offline RL.
- [x] **Preference Alignment**: GRPO for preference-based RL.
- [x] **Protein (Auto-Tune)**: Native Bayesian HPO system.

## Track 4: Multi-Backend & Portability (DONE)
*Objective: High throughput and cross-platform deployment.*

- [x] **no_std Core**: Complete refactor of core traits for embedded/WASM support.
- [x] **Multi-Backend**: Support for Torch, Burn, Candle, Luminal, and ONNX.
- [x] **HuggingFace Hub**: Integrated model distribution pipeline.
- [x] **Zero-Copy Batching**: High-performance observation piping (incl. Windows Shared Memory).

## 🛤️ Milestone 5: Research Frontiers (DONE)
*Objective: State-of-the-art exploration and safety.*

- [x] **Advanced Exploration**: Intrinsic Curiosity Module (ICM) and RND.
- [x] **Alignment**: RLHF/RLAIF toolkit with Bradley-Terry Reward Modeling.
- [x] **Formal Methods**: Runtime Action Shielding and Verified Policy traits.
- [x] **World Models**: Full implementation of RSSM and Dreamer-inspired planning.

---

## ⚡ Productivity Protections
- **Pre-commit Checklist**: `fmt` -> `clippy` -> `test` -> `bench`.
- **Atomic Commits**: No feature-bloat commits.
- **Benchmark Driven**: Every core Change must be accompanied by a `cargo bench` report.
