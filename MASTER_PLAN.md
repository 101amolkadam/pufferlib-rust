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

## 🛤️ Milestone 2: The Advanced Emulation Layer
*Objective: Full compatibility with complex simulations.*

- [ ] **Recursive Flattening**: Implement `SpaceTree` for deep `Dict/Tuple` support.
- [ ] **Agent Population**: Implement `max_agents` padding and masking for MARL parity.
- [ ] **Zero-Copy Unflattener**: High-speed structure restoration for policy networks.

## 🛤️ Milestone 3: Hardware Expansion & Parallelism
*Objective: Push throughput and ease of deployment.*

- [ ] **Staggered Batching**: Implement the Non-blocking `EnvPool-Lite` mechanism.
- [ ] **Candle Backend**: Enable pure-Rust tensor operations (optional alternative to LibTorch).
- [ ] **Shared Atomic Synchronization**: Eliminate OS mutexes in the vectorization hot-path.

## 🛤️ Milestone 4: Ecosystem & Algorithmic Parity
*Objective: Industry-standard feature set.*

- [ ] **Dual-Clip PPO**: Port PufferLib's specialized PPO stability features.
- [ ] **Self-Play Wrapper**: Native support for ELO-based training.
- [ ] **Gymnasium Bridge**: standardized wrappers for existing Rust environments.

---

## ⚡ Productivity Protections
- **Pre-commit Checklist**: `fmt` -> `clippy` -> `test` -> `bench`.
- **Atomic Commits**: No feature-bloat commits.
- **Benchmark Driven**: Every core Change must be accompanied by a `cargo bench` report.
