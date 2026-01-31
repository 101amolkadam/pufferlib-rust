# 🧪 Testing Strategy: PufferLib Rust

*Comprehensive testing plan to achieve 85%+ coverage for v1.0.0.*

---

## Overview

This document defines the testing strategy, tooling, and roadmap for achieving production-grade test coverage in PufferLib Rust.

---

## Current State

| Metric | Current | Target | Gap |
|:-------|:-------:|:------:|:---:|
| Overall Coverage | ~65% | 85% | -20% |
| Core (`env/`, `spaces/`) | ~80% | 95% | -15% |
| Algorithms (`training/`) | ~60% | 85% | -25% |
| Policies (`policy/`) | ~50% | 85% | -35% |

---

## Testing Pyramid

```
        ╱╲  E2E Tests (5%)
       ╱──╲  - Full training loops
      ╱────╲  - CLI command tests
     ╱──────╲
    ╱ Integration (20%) ╲  - Multi-component tests
   ╱────────────────────╲  - Algorithm correctness
  ╱ Unit Tests (75%)     ╲  - Individual functions
 ╱────────────────────────╲  - Edge cases, error paths
```

---

## Test Categories

### 1. Unit Tests

**Location:** Inline in source files or `src/<module>/tests.rs`

**Purpose:** Test individual functions and methods in isolation.

**Examples:**
```rust
// crates/pufferlib/src/spaces/flatten.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatten_box_space() {
        let space = DynSpace::Box { low: 0.0, high: 1.0, shape: vec![4] };
        let obs = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let flat = flatten(&space, &obs);
        assert_eq!(flat.len(), 4);
    }

    #[test]
    fn test_flatten_nested_dict() {
        let space = DynSpace::Dict(HashMap::from([
            ("a".to_string(), DynSpace::Discrete(2)),
            ("b".to_string(), DynSpace::Box { low: 0.0, high: 1.0, shape: vec![3] }),
        ]));
        // Test flattening...
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn test_flatten_invalid_shape() {
        // Test error case...
    }
}
```

**Coverage Targets by Module:**

| Module | Current | Target | Priority |
|:-------|:-------:|:------:|:--------:|
| `spaces/` | 75% | 95% | 🔴 High |
| `env/` | 80% | 95% | Medium |
| `vector/` | 70% | 90% | Medium |
| `policy/` | 50% | 85% | 🔴 High |
| `training/` | 60% | 85% | 🔴 High |
| `checkpoint/` | 70% | 90% | Medium |

---

### 2. Integration Tests

**Location:** `crates/pufferlib/tests/` and `tests/`

**Purpose:** Test interactions between multiple components.

**Examples:**
```rust
// tests/training_integration.rs
#[test]
fn test_ppo_training_loop() {
    let config = TrainerConfig::default()
        .timesteps(1000)
        .num_envs(4);
    
    let env_factory = || CartPole::new(42);
    let trainer = Trainer::new(config, env_factory).unwrap();
    
    let result = trainer.run();
    assert!(result.is_ok());
    
    let metrics = result.unwrap();
    assert!(metrics.mean_reward > 0.0);
}

#[test]
fn test_mappo_multi_agent() {
    // Test MAPPO with multiple agents...
}

#[test]
fn test_checkpoint_roundtrip() {
    // Save and load trainer state...
}
```

**Required Integration Tests:**

| Test | Components | Status |
|:-----|:-----------|:------:|
| PPO Full Loop | Trainer + Buffer + Policy + VecEnv | ⚠️ Partial |
| MAPPO Training | MappoTrainer + CentralizedCritic | ⚠️ Partial |
| DT Sequence | DecisionTransformer + ReplayDataset | ❌ Missing |
| DreamerV3 Imagination | WorldModel + RSSM + MPC | ❌ Missing |
| Checkpoint Save/Load | Trainer + Checkpoint | ✅ Done |
| Backend Parity | Torch vs Candle same output | ❌ Missing |

---

### 3. End-to-End Tests

**Location:** `tests/e2e/` or CLI tests

**Purpose:** Validate complete user workflows.

**Examples:**
```rust
// tests/cli_e2e.rs
use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn test_cli_train_cartpole() {
    let mut cmd = Command::cargo_bin("puffer").unwrap();
    cmd.args(["train", "cartpole", "--timesteps", "1000"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Epoch"));
}

#[test]
fn test_cli_eval_with_checkpoint() {
    // Train, save checkpoint, then evaluate...
}

#[test]
fn test_cli_list_environments() {
    let mut cmd = Command::cargo_bin("puffer").unwrap();
    cmd.args(["list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("cartpole"));
}
```

---

### 4. Property-Based Tests

**Tooling:** `proptest` crate

**Purpose:** Generate random inputs to find edge cases.

```rust
// tests/property_tests.rs
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_flatten_unflatten_roundtrip(
        values in prop::collection::vec(0.0f32..1.0, 1..100)
    ) {
        let space = DynSpace::Box { 
            low: 0.0, 
            high: 1.0, 
            shape: vec![values.len() as i64] 
        };
        let obs = Array1::from_vec(values.clone());
        let flat = flatten(&space, &obs);
        let restored = unflatten(&space, &flat);
        prop_assert_eq!(obs, restored);
    }

    #[test]
    fn test_gae_always_finite(
        rewards in prop::collection::vec(-10.0f32..10.0, 1..1000),
        values in prop::collection::vec(-100.0f32..100.0, 1..1001),
        gamma in 0.9f32..0.999,
        lambda in 0.9f32..0.999,
    ) {
        let advantages = compute_gae(&rewards, &values, gamma, lambda);
        for adv in advantages.iter() {
            prop_assert!(adv.is_finite(), "GAE produced non-finite value");
        }
    }
}
```

---

### 5. Fuzzing Tests

**Tooling:** `cargo-fuzz` with libFuzzer

**Purpose:** Find crashes and undefined behavior.

```rust
// fuzz/fuzz_targets/fuzz_deserialize.rs
#![no_main]
use libfuzzer_sys::fuzz_target;
use pufferlib::checkpoint::TrainerState;

fuzz_target!(|data: &[u8]| {
    // Ensure deserialization doesn't panic on arbitrary input
    let _ = bincode::deserialize::<TrainerState>(data);
});
```

**Setup:**
```bash
# Install
cargo install cargo-fuzz

# Run fuzzer
cargo +nightly fuzz run fuzz_deserialize
```

---

### 6. Performance Regression Tests

**Tooling:** `criterion` with CI integration

**Purpose:** Detect performance regressions before merge.

```rust
// benches/vectorization.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_vecenv_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("vecenv_step");
    
    for num_envs in [4, 16, 64, 256].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_envs),
            num_envs,
            |b, &num_envs| {
                let mut vec_env = Parallel::new(num_envs, || CartPole::new(42));
                let actions: Vec<i32> = vec![0; num_envs];
                
                b.iter(|| vec_env.step(&actions));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_vecenv_step);
criterion_main!(benches);
```

**CI Integration:**
```yaml
# .github/workflows/bench.yml
- name: Run benchmarks
  run: cargo bench --features torch -- --save-baseline main
  
- name: Compare benchmarks
  run: cargo bench --features torch -- --baseline main --threshold 10
```

---

## Testing Tooling

| Tool | Purpose | Command |
|:-----|:--------|:--------|
| `cargo test` | Run all tests | `cargo test --features torch` |
| `cargo-tarpaulin` | Coverage | `cargo tarpaulin --features torch` |
| `proptest` | Property testing | Included in test suite |
| `cargo-fuzz` | Fuzzing | `cargo +nightly fuzz run <target>` |
| `criterion` | Benchmarking | `cargo bench` |
| `assert_cmd` | CLI testing | Included in E2E tests |
| `miri` | Unsafe validation | `cargo +nightly miri test` |

---

## Coverage Measurement

### Running Coverage

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Run with HTML report
cargo tarpaulin --features torch --out Html --output-dir coverage/

# Run with Codecov format (CI)
cargo tarpaulin --features torch --out Xml --output-dir coverage/
```

### CI Configuration

```yaml
# .github/workflows/coverage.yml
name: Coverage

on: [push, pull_request]

jobs:
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install tarpaulin
        run: cargo install cargo-tarpaulin
        
      - name: Run coverage
        run: cargo tarpaulin --features torch --out Xml
        
      - name: Upload to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: cobertura.xml
          fail_ci_if_error: true
          minimum_coverage: 80
```

---

## Test Writing Guidelines

### Naming Conventions

```rust
// Pattern: test_<unit_under_test>_<scenario>_<expected_result>
#[test]
fn test_gae_single_step_returns_reward() { ... }

#[test]
fn test_flatten_empty_dict_returns_empty_array() { ... }

#[test]
#[should_panic(expected = "out of bounds")]
fn test_step_invalid_action_panics() { ... }
```

### Test Structure (AAA)

```rust
#[test]
fn test_example() {
    // Arrange - Set up test fixtures
    let env = CartPole::new(42);
    let action = 1;
    
    // Act - Execute the code under test
    let result = env.step(action);
    
    // Assert - Verify expectations
    assert!(result.reward >= 0.0);
    assert_eq!(result.observation.len(), 4);
}
```

---

## Priority Implementation Plan

### Phase 1 (Weeks 1-2): Critical Coverage
- [ ] Add tests for `policy/mlp.rs` forward/backward
- [ ] Add tests for `training/ppo.rs` loss computation
- [ ] Add tests for `training/gae.rs` edge cases
- [ ] Achieve 70% overall coverage

### Phase 2 (Weeks 3-4): Algorithm Coverage
- [ ] MAPPO integration tests
- [ ] GRPO integration tests
- [ ] DT offline training tests
- [ ] Achieve 75% overall coverage

### Phase 3 (Weeks 5-6): Property & Fuzz
- [ ] Property tests for space flattening
- [ ] Fuzzing for checkpoint deserialization
- [ ] Fuzzing for space parsing
- [ ] Achieve 80% overall coverage

### Phase 4 (Weeks 7-8): E2E & Polish
- [ ] CLI E2E tests for all commands
- [ ] Backend parity tests
- [ ] Performance regression baselines
- [ ] Achieve 85%+ overall coverage

---

## Test Data Management

### Test Fixtures

```
tests/
├── fixtures/
│   ├── checkpoints/
│   │   ├── v1_checkpoint.pt
│   │   └── corrupted_checkpoint.pt
│   ├── configs/
│   │   └── test_config.toml
│   └── trajectories/
│       └── cartpole_expert.json
```

### Loading Fixtures

```rust
use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures");
    path.push(name);
    path
}

#[test]
fn test_load_checkpoint_v1() {
    let path = fixture_path("checkpoints/v1_checkpoint.pt");
    let state = TrainerState::load(&path).unwrap();
    assert_eq!(state.version, 1);
}
```

---

*Document version: 1.0*
*Last updated: 2026-01-31*
