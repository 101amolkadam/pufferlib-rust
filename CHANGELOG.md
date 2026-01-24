# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Real-time throughput dashboard (TUI) with `indicatif` progress bars.
- CLI Demos for `Memory` and `Squared` environments with specialized rendering.
- `SmallVec` integration for zero-allocation environment metrics.
- Native `sample()` implementation for `Dict` observation/action spaces.

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
