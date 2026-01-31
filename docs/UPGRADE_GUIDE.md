# 🔄 Upgrade Guide: PufferLib Rust

*Step-by-step guide for upgrading PufferLib Rust between versions.*

---

## Overview

This guide provides migration instructions, breaking changes, and upgrade procedures for software engineers updating PufferLib Rust.

---

## Version History

| Version | Release Date | Key Changes |
|:--------|:------------:|:------------|
| v0.1.0 | 2026-01-24 | Initial release with core PPO implementation |
| v0.2.0 | Q2 2026 (Planned) | API stabilization, enhanced coverage |
| v1.0.0 | Q4 2026 (Planned) | Stable API, production-ready |

---

## Upgrading to v0.2.0 (Planned)

### Breaking Changes

> [!CAUTION]
> The following changes require code modifications when upgrading from v0.1.0 to v0.2.0.

#### 1. `Policy` Trait Signature Change

**Before (v0.1.0):**
```rust
pub trait Policy {
    fn forward(&self, obs: &Tensor) -> (Tensor, Tensor);
    fn act(&self, obs: &Tensor, deterministic: bool) -> Tensor;
}
```

**After (v0.2.0):**
```rust
pub trait Policy {
    type Tensor: Clone + Send;
    
    fn forward(&self, obs: &Self::Tensor) -> PolicyOutput<Self::Tensor>;
    fn act(&self, obs: &Self::Tensor, mode: ActionMode) -> Self::Tensor;
}

pub enum ActionMode {
    Stochastic,
    Deterministic,
    Mode,  // Take argmax for discrete, mean for continuous
}
```

**Migration:**
```rust
// Old code
let (action, value) = policy.forward(&obs);
let action = policy.act(&obs, true);

// New code
use pufferlib::policy::{Policy, ActionMode, PolicyOutput};
let PolicyOutput { action, value, .. } = policy.forward(&obs);
let action = policy.act(&obs, ActionMode::Deterministic);
```

#### 2. Checkpoint Format Version Bump

**Impact:** Old checkpoint files are incompatible with v0.2.0.

**Migration:**
```bash
# Option 1: Re-train from scratch
cargo run --release --bin puffer -- train cartpole --timesteps 1000000

# Option 2: Use migration tool (if provided)
cargo run --release --bin puffer -- migrate-checkpoint --from v0.1.0 --input old_checkpoint.pt --output new_checkpoint.pt
```

#### 3. Feature Flag Renames

| Old (v0.1.0) | New (v0.2.0) | Reason |
|:-------------|:-------------|:-------|
| `torch` | `backend-torch` | Backend naming consistency |
| `candle` | `backend-candle` | Backend naming consistency |
| `burn` | `backend-burn` | Backend naming consistency |

**Migration:**
```toml
# Old Cargo.toml
[dependencies]
pufferlib = { version = "0.1", features = ["torch"] }

# New Cargo.toml
[dependencies]
pufferlib = { version = "0.2", features = ["backend-torch"] }
```

### Deprecations

The following APIs are deprecated in v0.2.0 and will be removed in v1.0.0:

| Deprecated | Replacement | Notes |
|:-----------|:------------|:------|
| `Trainer::train()` | `Trainer::run()` | More descriptive name |
| `ExperienceBuffer::clear()` | `ExperienceBuffer::reset()` | Semantic clarity |
| `MetricLogger::log_scalar()` | `MetricLogger::record()` | Unified API |

### New Features

- **API Stability Annotations**: Public types now have stability markers
- **Improved Error Messages**: `anyhow` context for all errors
- **Backend Abstraction**: Unified `Backend` trait for multi-backend policies

---

## Upgrading to v1.0.0 (Planned)

### Semantic Versioning Commitment

Starting with v1.0.0, PufferLib Rust follows strict [semantic versioning](https://semver.org/):
- **Major** (1.x.x): Breaking API changes
- **Minor** (x.1.x): New features, backward compatible
- **Patch** (x.x.1): Bug fixes only

### Breaking Changes (from v0.2.0)

> [!WARNING]
> These changes are planned but not finalized. Check release notes for actual changes.

#### 1. Removed Deprecated APIs

All v0.2.0 deprecations are removed:
```rust
// These no longer compile in v1.0.0
trainer.train();           // Use: trainer.run()
buffer.clear();            // Use: buffer.reset()
logger.log_scalar(k, v);   // Use: logger.record(k, v)
```

#### 2. `PufferEnv` Trait Stabilization

**Final signature:**
```rust
pub trait PufferEnv: Send + 'static {
    type Observation: Clone + Send;
    type Action: Clone + Send;
    
    fn reset(&mut self, options: ResetOptions) -> Self::Observation;
    fn step(&mut self, action: Self::Action) -> StepResult<Self::Observation>;
    fn render(&self, mode: RenderMode) -> Option<RenderOutput>;
    fn close(&mut self);
    
    fn observation_space(&self) -> Space;
    fn action_space(&self) -> Space;
    fn metadata(&self) -> EnvMetadata;
}
```

#### 3. Configuration File Format

**v0.x (TOML):**
```toml
[training]
learning_rate = 0.0003
gamma = 0.99
```

**v1.0.0 (YAML with schema validation):**
```yaml
version: "1.0"
training:
  learning_rate: 0.0003
  gamma: 0.99
```

### Migration Checklist

- [ ] Update `Cargo.toml` dependencies
- [ ] Run `cargo clippy` to find deprecated API usage
- [ ] Update all deprecated API calls
- [ ] Convert configuration files to new format
- [ ] Re-run test suite
- [ ] Verify checkpoint compatibility or migrate
- [ ] Update CI/CD scripts

---

## Version Compatibility Matrix

### Backend Versions

| PufferLib | LibTorch | Candle | Burn | Rust MSRV |
|:----------|:---------|:-------|:-----|:----------|
| 0.1.x | 2.0-2.2 | 0.8.x | 0.13.x | 1.70 |
| 0.2.x | 2.2+ | 0.9.x | 0.14.x | 1.75 |
| 1.0.x | 2.4+ | 0.10.x | 0.15.x | 1.78 |

### Checkpoint Compatibility

| Checkpoint Format | Compatible Versions |
|:------------------|:-------------------|
| Format v1 | 0.1.x only |
| Format v2 | 0.2.x, 1.0.x |

---

## Feature Migration Paths

### From Python PufferLib to Rust

**Environment Migration:**

```python
# Python
class MyEnv(gymnasium.Env):
    def __init__(self):
        self.observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(4,))
        self.action_space = gymnasium.spaces.Discrete(2)
    
    def reset(self, seed=None):
        return np.zeros(4), {}
    
    def step(self, action):
        return np.zeros(4), 0.0, False, False, {}
```

```rust
// Rust equivalent
use pufferlib::prelude::*;

pub struct MyEnv { ... }

impl PufferEnv for MyEnv {
    type Observation = Array1<f32>;
    type Action = i32;
    
    fn observation_space(&self) -> DynSpace {
        DynSpace::Box {
            low: -1.0,
            high: 1.0,
            shape: vec![4],
        }
    }
    
    fn action_space(&self) -> DynSpace {
        DynSpace::Discrete(2)
    }
    
    fn reset(&mut self, _options: ResetOptions) -> Self::Observation {
        Array1::zeros(4)
    }
    
    fn step(&mut self, _action: Self::Action) -> StepResult<Self::Observation> {
        StepResult {
            observation: Array1::zeros(4),
            reward: 0.0,
            terminated: false,
            truncated: false,
            cost: 0.0,
            info: EnvInfo::default(),
        }
    }
}
```

### From Stable Baselines 3

**Training Script Migration:**

```python
# SB3
from stable_baselines3 import PPO
model = PPO("MlpPolicy", "CartPole-v1", verbose=1)
model.learn(total_timesteps=100000)
```

```rust
// PufferLib Rust equivalent
use pufferlib::prelude::*;
use pufferlib_envs::CartPole;

let config = TrainerConfig::default()
    .timesteps(100000)
    .learning_rate(3e-4);

let env_factory = || CartPole::new(42);
let trainer = Trainer::new(config, env_factory)?;
trainer.run()?;
```

---

## Troubleshooting Upgrades

### Common Issues

| Issue | Solution |
|:------|:---------|
| `trait bound not satisfied` after upgrade | Check for renamed traits or methods |
| Checkpoint load fails | Use migration tool or retrain |
| Feature not found | Check renamed feature flags |
| MSRV error | Update Rust toolchain |

### Getting Help

1. Check [CHANGELOG.md](../CHANGELOG.md) for complete change list
2. Search [GitHub Issues](https://github.com/101amolkadam/pufferlib-rust/issues)
3. Ask on [PufferAI Discord](https://discord.gg/puffer)

---

## Rollback Procedure

If an upgrade fails:

```bash
# 1. Pin to previous version
# Cargo.toml
[dependencies]
pufferlib = "=0.1.0"

# 2. Clear build artifacts
cargo clean

# 3. Rebuild
cargo build --release --features torch

# 4. Verify
cargo test --features torch
```

---

*Document version: 1.0*
*Last updated: 2026-01-31*
