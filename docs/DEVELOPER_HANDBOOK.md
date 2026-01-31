# 📚 PufferLib Rust Developer Handbook

*Practical guide for software engineers working on PufferLib Rust.*

---

## 📖 Table of Contents
1. [Development Environment Setup](#development-environment-setup)
2. [Project Structure](#project-structure)
3. [Building the Project](#building-the-project)
4. [Testing Guide](#testing-guide)
5. [Code Style & Conventions](#code-style--conventions)
6. [Common Development Tasks](#common-development-tasks)
7. [Debugging Guide](#debugging-guide)
8. [Performance Profiling](#performance-profiling)
9. [Release Process](#release-process)

---

## Development Environment Setup

### Prerequisites

| Tool | Version | Purpose |
|:-----|:--------|:--------|
| Rust | 1.70+ (stable) | Core language |
| LibTorch | 2.x | PyTorch C++ backend |
| Protobuf | 3.x | gRPC protocol buffers |
| CMake | 3.18+ | Native builds |

### Windows Setup

```powershell
# 1. Install Rust
winget install -e --id Rustlang.Rustup
rustup default stable

# 2. Install LibTorch (uses VERSIONS file)
.\setup_libtorch.ps1

# 3. Install Protobuf
winget install -e --id ProtocolBuffers.Protoc

# 4. Verify installation
cargo check --features torch
```

### Linux/macOS Setup

```bash
# 1. Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 2. Install LibTorch
# Option A: Download and extract manually
# Option B: Use system package manager (apt/brew)

# 3. Set environment variables
export LIBTORCH=/path/to/libtorch
export LIBTORCH_USE_PYTORCH=1

# 4. Install Protobuf
# Ubuntu: sudo apt install protobuf-compiler
# macOS: brew install protobuf

# 5. Verify
cargo check --features torch
```

### IDE Configuration

**VSCode (Recommended)**
```json
// .vscode/settings.json
{
  "rust-analyzer.cargo.features": ["torch"],
  "rust-analyzer.checkOnSave.command": "clippy"
}
```

**JetBrains RustRover/CLion**
- Enable Clippy lints in Settings → Languages & Frameworks → Rust → External Linters

---

## Project Structure

```
pufferlib-rust/
├── Cargo.toml              # Workspace manifest
├── VERSIONS                 # LibTorch/CUDA version config
├── crates/
│   ├── pufferlib/          # Core library (main crate)
│   │   ├── src/
│   │   │   ├── env/        # Environment traits
│   │   │   ├── spaces/     # Observation/action spaces
│   │   │   ├── vector/     # Vectorization backends
│   │   │   ├── policy/     # Neural network policies
│   │   │   ├── training/   # Training algorithms
│   │   │   ├── mappo/      # Multi-agent PPO
│   │   │   ├── grpo/       # Group Relative Policy Optimization
│   │   │   ├── dapo/       # Decoupled APPO
│   │   │   ├── dreamer/    # World models
│   │   │   ├── offline/    # Offline RL & Decision Transformer
│   │   │   ├── checkpoint/ # State persistence
│   │   │   ├── log/        # Logging (TensorBoard, W&B)
│   │   │   └── utils/      # Utilities
│   │   ├── examples/       # Example scripts
│   │   └── tests/          # Integration tests
│   ├── pufferlib-envs/     # Environment implementations
│   ├── pufferlib-cli/      # Command-line interface
│   ├── pufferlib-bevy/     # Bevy game engine integration
│   ├── pufferlib-python/   # PyO3 Python bindings
│   ├── pufferlib-rpc/      # gRPC distributed training
│   └── pufferlib-wasm/     # WebAssembly module
├── docs/                   # Documentation
│   └── v2/                 # V2 phase documentation
├── tests/                  # Workspace-level tests
└── benches/               # Benchmarks
```

---

## Building the Project

### Feature Flags

| Flag | Description | Default |
|:-----|:------------|:-------:|
| `std` | Standard library support | ✅ On |
| `torch` | LibTorch backend | Off |
| `candle` | Candle pure-Rust backend | Off |
| `burn` | Burn ML framework backend | Off |
| `luminal` | Luminal graph compiler | Off |
| `onnx` | ONNX Runtime export | Off |
| `checkpoint` | State serialization | Off |
| `checkpoint-compressed` | Compressed checkpoints | Off |
| `tensorboard` | TensorBoard logging | Off |
| `python` | Python bindings | Off |

### Common Build Commands

```bash
# Development build (fast compile, no optimizations)
cargo build --features torch

# Release build (optimized)
cargo build --release --features torch

# Check without building
cargo check --features torch

# Build all crates in workspace
cargo build --workspace --features torch

# Build specific crate
cargo build -p pufferlib-cli --features torch

# Build with multiple features
cargo build --features "torch,tensorboard,checkpoint-compressed"

# Build without Torch (pure Rust with Candle)
cargo build --features candle
```

### Running Examples

```bash
# Run CartPole training
cargo run --release --bin puffer --features torch -- train cartpole --timesteps 100000

# Run with environment variables for LibTorch
$env:LIBTORCH = "D:\libtorch"; cargo run --release --features torch --example repro_panic

# Run profiling example
cargo run --release --features torch --example profile_training
```

---

## Testing Guide

### Running Tests

```bash
# Run all tests
cargo test --features torch

# Run tests for specific crate
cargo test -p pufferlib --features torch

# Run specific test
cargo test --features torch -- test_mappo

# Run tests with output
cargo test --features torch -- --nocapture

# Run ignored (slow) tests
cargo test --features torch -- --ignored

# Run doc tests
cargo test --doc --features torch
```

### Test Categories

| Category | Location | Command |
|:---------|:---------|:--------|
| Unit tests | `src/**/tests.rs` or inline | `cargo test` |
| Integration tests | `tests/` | `cargo test --test '*'` |
| Doc tests | Inline `///` docs | `cargo test --doc` |
| Examples (as tests) | `examples/` | `cargo test --examples` |

### Writing Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        // Arrange
        let env = CartPole::new(42);
        
        // Act
        let obs = env.reset(None);
        
        // Assert
        assert_eq!(obs.len(), 4);
    }

    #[test]
    #[ignore] // Slow test, run with --ignored
    fn test_full_training_loop() {
        // ...
    }
}
```

### Coverage

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Run coverage
cargo tarpaulin --features torch --out Html

# View report
open tarpaulin-report.html
```

---

## Code Style & Conventions

### Formatting & Linting

```bash
# Format code
cargo fmt

# Check formatting (CI)
cargo fmt -- --check

# Run Clippy
cargo clippy --features torch -- -D warnings

# Pre-commit checklist (run before every commit)
cargo fmt && cargo clippy --features torch -- -D warnings && cargo test --features torch
```

### Naming Conventions

| Item | Convention | Example |
|:-----|:-----------|:--------|
| Modules | snake_case | `checkpoint`, `grpo` |
| Structs | PascalCase | `ExperienceBuffer`, `MlpPolicy` |
| Traits | PascalCase | `PufferEnv`, `VecEnv` |
| Functions | snake_case | `compute_gae`, `step_envs` |
| Constants | SCREAMING_SNAKE | `MAX_AGENTS`, `DEFAULT_GAMMA` |
| Feature flags | kebab-case | `checkpoint-compressed` |

### Code Organization

```rust
// File structure order:
// 1. Module documentation
//! This module implements PPO utilities.

// 2. Imports (std, external, internal)
use std::collections::HashMap;
use ndarray::Array1;
use crate::spaces::DynSpace;

// 3. Constants
const DEFAULT_CLIP_RANGE: f32 = 0.2;

// 4. Types and traits
pub trait Policy { ... }

// 5. Struct definitions
pub struct Trainer { ... }

// 6. Impl blocks (inherent, then trait)
impl Trainer { ... }
impl Clone for Trainer { ... }

// 7. Private helpers
fn compute_advantage(...) { ... }

// 8. Tests
#[cfg(test)]
mod tests { ... }
```

---

## Common Development Tasks

### Adding a New Algorithm

1. **Create module**: `crates/pufferlib/src/<algorithm>/mod.rs`
2. **Define config**:
   ```rust
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct MyAlgoConfig {
       pub learning_rate: f64,
       pub gamma: f64,
       // ...
   }
   ```
3. **Implement core logic**: Loss computation, update steps
4. **Integrate with Trainer**: Add to training loop
5. **Add tests**: `tests/<algorithm>_test.rs`
6. **Update docs**: Add to relevant documentation

### Adding a New Environment

1. **Create file**: `crates/pufferlib-envs/src/<env>.rs`
2. **Implement `PufferEnv`**:
   ```rust
   impl PufferEnv for MyEnv {
       type Observation = Array1<f32>;
       type Action = i32;
       
       fn reset(&mut self, seed: Option<u64>) -> Self::Observation { ... }
       fn step(&mut self, action: Self::Action) -> StepResult<Self::Observation> { ... }
       fn observation_space(&self) -> DynSpace { ... }
       fn action_space(&self) -> DynSpace { ... }
   }
   ```
3. **Add to CLI**: Register in `pufferlib-cli/src/envs.rs`
4. **Add tests**: Unit tests for reset/step

### Adding a New Backend

1. **Create module**: `crates/pufferlib/src/policy/<backend>.rs`
2. **Implement policy**:
   ```rust
   pub struct MyBackendMlp { ... }
   
   impl MyBackendMlp {
       pub fn forward(&self, obs: &Tensor) -> (Tensor, Tensor) { ... }
   }
   ```
3. **Feature gate**: Add to `Cargo.toml`
4. **Add parity tests**: Same input → same output

---

## Debugging Guide

### Common Issues

| Issue | Cause | Solution |
|:------|:------|:---------|
| LibTorch not found | `LIBTORCH` not set | Run `setup_libtorch.ps1` |
| CUDA out of memory | Batch size too large | Reduce `num_envs` or `batch_size` |
| Panic in training | NaN in loss | Check learning rate, add gradient clipping |
| Slow compilation | Too many features | Use `--no-default-features` |

### Debug Logging

```rust
use tracing::{debug, info, warn, error, trace};

// Enable in code
tracing_subscriber::fmt()
    .with_env_filter("pufferlib=debug")
    .init();

// Or via environment
$env:RUST_LOG = "pufferlib=debug"; cargo run ...
```

### Debugging with LLDB/GDB

```bash
# Build with debug symbols
cargo build --features torch

# Debug
rust-lldb target/debug/puffer
# or
rust-gdb target/debug/puffer
```

---

## Performance Profiling

### CPU Profiling

```bash
# Install flamegraph
cargo install flamegraph

# Profile training
cargo flamegraph --features torch --bin puffer -- train cartpole --timesteps 10000
```

### Memory Profiling

```bash
# Install heaptrack (Linux)
sudo apt install heaptrack

# Profile
heaptrack cargo run --release --features torch -- train cartpole --timesteps 10000
heaptrack_gui heaptrack.*.gz
```

### Benchmark Suite

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench vectorization
```

---

## Release Process

### Version Bump

```bash
# Update version in all Cargo.toml files
# Workspace version in root Cargo.toml

# Update CHANGELOG.md
# Update documentation dates

# Tag release
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
```

### Pre-release Checklist

- [ ] All tests pass: `cargo test --features torch`
- [ ] No Clippy warnings: `cargo clippy --features torch -- -D warnings`
- [ ] Code formatted: `cargo fmt -- --check`
- [ ] Docs build: `cargo doc --features torch`
- [ ] CHANGELOG updated
- [ ] README updated
- [ ] Version bumped

---

## Quick Reference

### Essential Commands

```bash
# Build + test + lint (do before commits)
cargo fmt && cargo clippy --features torch -- -D warnings && cargo test --features torch

# Clean build
cargo clean && cargo build --release --features torch

# Profile training
cargo run --release --features torch --bin puffer -- train cartpole --timesteps 100000

# Generate docs
cargo doc --features torch --open
```

### Environment Variables

| Variable | Purpose |
|:---------|:--------|
| `LIBTORCH` | LibTorch installation path |
| `LIBTORCH_USE_PYTORCH` | Use PyTorch installation |
| `CUDA_VISIBLE_DEVICES` | GPU selection |
| `RUST_LOG` | Logging level |
| `RUST_BACKTRACE` | Panic backtraces (`1` or `full`) |

---

*Document version: 1.0*
*Last updated: 2026-01-31*
