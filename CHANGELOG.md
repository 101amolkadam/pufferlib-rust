# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **no_std Core Support**: Refactored core traits (`PufferEnv`, `DynSpace`) for embedded and WASM compatibility.
- **Multi-Backend Architecture**: Added support for **Burn**, **Candle**, **Luminal**, and **ONNX (Ort)** tensor engines.
- **Advanced Exploration**: Implemented Intrinsic Curiosity Module (**ICM**) and Random Network Distillation (**RND**).
- **Goal-Conditioned RL**: Added **Hindsight Experience Replay (HER)** trajectory relabeling.
- **RLHF/RLAIF Toolkit**: New reward modeling and **Constitutional AI** alignment modules.
- **Formal Methods**: Added action **Shielding** and **Verified Policy** traits for safety-critical RL.
- **HuggingFace Hub Integration**: Direct model upload/download support via `push_to_hub`.
- **Custom Metrics Callbacks**: Unified system for user-defined metrics during training.
- **Automatic Mixed Precision (AMP)** and **Gradient Accumulation** for high-performance training.

### Changed
- Decoupled `Trainer` logic to support pluggable backends (Torch primary, others in progress).
- Replaced standard library types with abstracted `types` module to support `no_std` targets.
- Updated `ort` dependency to `2.0.0-rc.11` for enhanced ONNX support.

### Changed
- Refactored `Parallel` backend to be lock-free (removed `Arc<Mutex<E>>`) for extreme throughput.
- Monomorphized `VecEnv` and `PufferEnv` wrappers for zero-overhead static dispatch.
- Optimized observation and action data flow to reduce heap allocations.
- Fixed GitHub Actions CI/CD pipeline (Clippy and formatting stability).

## [0.1.0] - 2026-01-24

### Added
- Initial Rust port of PufferLib core.
- Basic PPO training implementation.
- Support for CartPole, Bandit, Memory, and Squared environments.
- High-bandwidth environment vectorization.
