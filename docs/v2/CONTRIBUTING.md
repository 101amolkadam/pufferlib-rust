# Contributing to PufferLib v2

*Developer guide for implementing Roadmap 2.0 features.*

---

## Getting Started

### Prerequisites

```bash
# Core requirements
rustup update stable
cargo install cargo-nextest  # Faster test runner

# For torch feature
# Download LibTorch from https://pytorch.org/
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH

# For development
cargo install cargo-watch    # Auto-rebuild on changes
cargo install cargo-llvm-cov # Coverage reports
```

### Build & Test

```bash
# Build without ML backends (fast iteration)
cargo check -p pufferlib

# Build with LibTorch
cargo check -p pufferlib --features torch

# Build with Candle
cargo check -p pufferlib --features candle

# Run all tests
cargo nextest run --workspace

# Run specific test
cargo test -p pufferlib test_gae_computation
```

---

## Development Workflow

### Feature Branch Strategy

```
main
├── feature/phase5-decision-transformer
├── feature/phase5-mappo
├── feature/phase6-checkpointing
└── bugfix/trainer-memory-leak
```

### Commit Message Format

```
type(scope): description

[optional body]

[optional footer]
```

**Types**: `feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `chore`

**Examples**:
```
feat(training): add Decision Transformer policy
fix(ppo): correct KL divergence computation for continuous actions
docs(v2): add MAPPO implementation guide
```

---

## Code Standards

### Rust Style

- Follow `rustfmt` defaults (run `cargo fmt` before commit)
- Use `clippy` lints: `cargo clippy --all-targets --all-features`
- Document all public items with `///` doc comments
- Use `#[must_use]` for functions returning important values

### Error Handling

```rust
// Use the library's error type
use crate::{PufferError, Result};

pub fn do_something() -> Result<Tensor> {
    // Use ? for propagation
    let value = risky_operation()?;
    
    // Return specific errors when needed
    if value < 0.0 {
        return Err(PufferError::TrainingError(
            "Value must be non-negative".into()
        ));
    }
    
    Ok(compute(value))
}
```

### Testing Patterns

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Unit tests are fast and isolated
    #[test]
    fn test_basic_functionality() {
        let result = function_under_test(input);
        assert_eq!(result, expected);
    }

    // Integration tests need feature flags
    #[test]
    #[cfg(feature = "torch")]
    fn test_with_torch_backend() {
        // Uses real tensors
    }

    // Property-based testing for complex logic
    #[test]
    fn test_gae_properties() {
        // Advantages sum should relate to total return
    }
}
```

---

## Adding New Features

### Phase 5: New Algorithm Checklist

- [ ] Create module file (e.g., `src/training/mappo.rs`)
- [ ] Define structs with `#[derive(Clone, Debug)]`
- [ ] Implement core trait (e.g., `Policy`, `Trainer`)
- [ ] Add unit tests in the same file
- [ ] Add integration test in `tests/`
- [ ] Export from `mod.rs` with feature gate if needed
- [ ] Document with examples
- [ ] Update CHANGELOG.md

### Phase 6: Production Feature Checklist

- [ ] Design for fault tolerance (graceful degradation)
- [ ] Add configuration options to `TrainerConfig`
- [ ] Ensure backward compatibility
- [ ] Add CLI integration if applicable
- [ ] Document operational aspects (disk usage, network, etc.)
- [ ] Add monitoring/metrics hooks

### Phase 7: New Backend Checklist

- [ ] Create backend module (e.g., `src/backend/burn.rs`)
- [ ] Implement `Backend` trait
- [ ] Adapt at least `MlpPolicy` to new backend
- [ ] Add feature flag to `Cargo.toml`
- [ ] Ensure CI tests the new feature
- [ ] Document setup instructions
- [ ] Add performance benchmarks

---

## Performance Guidelines

### Tensor Operations

```rust
// BAD: Creates intermediate tensors
let result = tensor1.add(&tensor2).mul(&tensor3);

// GOOD: In-place when possible
let mut result = tensor1.shallow_clone();
result += &tensor2;
result *= &tensor3;
```

### Memory Management

```rust
// Detach tensors before storing to prevent graph retention
buffer.store_observation(obs.detach());

// Use no_grad for inference
let action = tch::no_grad(|| policy.forward(&obs));
```

### Profiling

```bash
# CPU profiling
cargo build --release --features torch
perf record ./target/release/pufferlib-cli train
perf report

# Memory profiling  
valgrind --tool=massif ./target/release/pufferlib-cli train
ms_print massif.out.*
```

---

## Documentation

### Module Documentation Template

```rust
//! # Module Name
//!
//! Brief description of what this module does.
//!
//! ## Overview
//!
//! Longer explanation of the module's purpose and design.
//!
//! ## Example
//!
//! ```rust,ignore
//! use pufferlib::module_name::Component;
//!
//! let component = Component::new(config);
//! let result = component.do_thing();
//! ```
//!
//! ## References
//!
//! - [Paper Name](https://arxiv.org/abs/xxx)
//! - [Related Documentation](link)
```

### Function Documentation Template

```rust
/// Brief one-line description.
///
/// Longer description if needed, explaining the function's
/// behavior, edge cases, and design rationale.
///
/// # Arguments
///
/// * `arg1` - Description of first argument
/// * `arg2` - Description of second argument
///
/// # Returns
///
/// Description of return value.
///
/// # Errors
///
/// Returns `PufferError::X` if condition Y.
///
/// # Example
///
/// ```rust,ignore
/// let result = function_name(arg1, arg2)?;
/// ```
pub fn function_name(arg1: Type1, arg2: Type2) -> Result<ReturnType> {
    // implementation
}
```

---

## Pull Request Process

1. **Self-review**: Run `cargo fmt`, `cargo clippy`, `cargo test`
2. **Description**: Explain what and why (link to issue if applicable)
3. **Checklist**:
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] CHANGELOG.md updated
   - [ ] No breaking changes (or documented in PR)
4. **Review**: Address feedback promptly
5. **Squash**: Keep commit history clean

---

## Getting Help

- **Architecture questions**: See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Algorithm details**: See `docs/v2/algorithms/`
- **Production concerns**: See `docs/v2/production/`
- **Discussion**: Open a GitHub Discussion

---

*Last updated: 2026-01-28*
