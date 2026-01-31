# ⚡ Performance Optimization Guide: PufferLib Rust

*Roadmap and techniques for achieving 2.5M+ steps per second.*

---

## Executive Summary

| Metric | Current | Target | Improvement Needed |
|:-------|:-------:|:------:|:------------------:|
| Steps/Second (CartPole, 8 cores) | ~1.8M | 2.5M | +39% |
| Memory per environment | ~4KB | ~2KB | -50% |
| Policy forward latency | ~50μs | ~30μs | -40% |

---

## Current Performance Profile

Based on profiling with `cargo flamegraph` and `coz`:

```
Performance Breakdown (100% = training loop)
├── Environment Stepping: 35%
│   ├── VecEnv dispatch: 3% (optimized)
│   ├── Env.step() logic: 25%
│   └── Observation copy: 7%
├── Policy Forward: 30%
│   ├── Tensor creation: 10%
│   ├── Neural network: 18%
│   └── Action sampling: 2%
├── Buffer Operations: 20%
│   ├── GAE computation: 12%
│   ├── Memory allocation: 5%
│   └── Advantage normalization: 3%
└── Optimizer Step: 15%
    ├── Gradient computation: 12%
    └── Weight update: 3%
```

---

## Optimization Strategies

### 1. SIMD-Accelerated Space Operations

**Current:** Scalar loops for flattening/unflattening
**Target:** SIMD vectorization using `std::simd` (nightly) or `packed_simd`

**Impact:** ~15% total improvement

**Implementation:**

```rust
// Before (scalar)
fn flatten_box(obs: &Array1<f32>, output: &mut [f32]) {
    for (i, val) in obs.iter().enumerate() {
        output[i] = *val;
    }
}

// After (SIMD)
#[cfg(target_feature = "avx2")]
fn flatten_box_simd(obs: &Array1<f32>, output: &mut [f32]) {
    use std::simd::f32x8;
    
    let chunks = obs.len() / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let simd = f32x8::from_slice(&obs.as_slice().unwrap()[offset..]);
        simd.copy_to_slice(&mut output[offset..offset + 8]);
    }
    // Handle remainder with scalar fallback
}
```

**Files to modify:**
- `crates/pufferlib/src/spaces/flatten.rs`
- `crates/pufferlib/src/spaces/unflatten.rs`

---

### 2. GAE SIMD Optimization

**Current:** Sequential advantage computation
**Target:** Vectorized GAE with SIMD

**Impact:** ~10% total improvement

**Implementation:**

```rust
// Before
pub fn compute_gae(
    rewards: &[f32],
    values: &[f32],
    gamma: f32,
    lambda: f32,
) -> Vec<f32> {
    let mut advantages = vec![0.0; rewards.len()];
    let mut last_gae = 0.0;
    
    for t in (0..rewards.len()).rev() {
        let delta = rewards[t] + gamma * values[t + 1] - values[t];
        last_gae = delta + gamma * lambda * last_gae;
        advantages[t] = last_gae;
    }
    advantages
}

// After (with SIMD batch processing)
pub fn compute_gae_batched(
    rewards: &Array2<f32>,  // [num_envs, timesteps]
    values: &Array2<f32>,
    gamma: f32,
    lambda: f32,
) -> Array2<f32> {
    // Process all environments in parallel using SIMD
    // Each environment's GAE computed with vectorized operations
}
```

**Files to modify:**
- `crates/pufferlib/src/training/gae.rs`

---

### 3. Memory Pre-allocation

**Current:** Dynamic allocation during rollout
**Target:** Pre-allocated arena for all buffers

**Impact:** ~5% total improvement

**Implementation:**

```rust
// Before
impl ExperienceBuffer {
    pub fn add(&mut self, obs: Array1<f32>, ...) {
        self.observations.push(obs); // Allocation on every call
    }
}

// After
pub struct PreallocatedBuffer {
    observations: Array2<f32>,  // Pre-allocated [capacity, obs_dim]
    write_idx: usize,
}

impl PreallocatedBuffer {
    pub fn new(capacity: usize, obs_dim: usize) -> Self {
        Self {
            observations: Array2::zeros((capacity, obs_dim)),
            write_idx: 0,
        }
    }
    
    pub fn add(&mut self, obs: ArrayView1<f32>) {
        self.observations.row_mut(self.write_idx).assign(&obs);
        self.write_idx += 1;
    }
}
```

**Files to modify:**
- `crates/pufferlib/src/training/buffer.rs`

---

### 4. Observation Zero-Copy Pipeline

**Current:** Observations copied from env → buffer → tensor
**Target:** Direct memory mapping from env to GPU

**Impact:** ~7% total improvement

**Implementation:**

```rust
// Create shared memory region
pub struct SharedObservationBuffer {
    #[cfg(target_os = "windows")]
    mapping: windows_sys::Win32::System::Memory::HANDLE,
    #[cfg(target_os = "linux")]
    mapping: *mut c_void,
    ptr: *mut f32,
    len: usize,
}

impl SharedObservationBuffer {
    pub fn as_tensor(&self) -> Tensor {
        // Create tensor view without copy
        unsafe {
            Tensor::from_data_size(
                self.ptr as *const f32,
                &[self.len as i64],
                Kind::Float,
            )
        }
    }
}
```

**Files to modify:**
- `crates/pufferlib/src/vector/shared_memory.rs`

---

### 5. Async Trainer Architecture

**Current:** Synchronous rollout → training → rollout cycle
**Target:** Pipelined async rollout with overlapped training

**Impact:** ~20% total improvement (CPU utilization)

**Architecture:**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Rollout    │───►│   Buffer    │───►│  Training   │
│  Thread     │    │   Queue     │    │  Thread     │
└─────────────┘    └─────────────┘    └─────────────┘
      │                                      │
      └──────── Updated Policy ◄─────────────┘
```

**Implementation:**

```rust
pub struct AsyncTrainer {
    rollout_handle: JoinHandle<()>,
    training_handle: JoinHandle<()>,
    buffer_tx: Sender<RolloutData>,
    buffer_rx: Receiver<RolloutData>,
    policy: Arc<RwLock<Policy>>,
}

impl AsyncTrainer {
    pub fn run(&mut self) {
        // Spawn rollout thread
        let policy = Arc::clone(&self.policy);
        let tx = self.buffer_tx.clone();
        
        thread::spawn(move || {
            loop {
                let policy_guard = policy.read().unwrap();
                let rollout = collect_rollout(&policy_guard, &mut envs);
                tx.send(rollout).unwrap();
            }
        });
        
        // Training loop
        for rollout in self.buffer_rx.iter() {
            let mut policy_guard = self.policy.write().unwrap();
            update_policy(&mut policy_guard, &rollout);
        }
    }
}
```

**Files to modify:**
- `crates/pufferlib/src/training/async_trainer.rs` (new)
- `crates/pufferlib/src/training/mod.rs`

---

### 6. GPU-Native Environments

**Future:** Run environment logic directly on GPU

**Impact:** Potential 10-100x improvement for suitable envs

**Concept:**

```rust
pub trait GpuEnv {
    fn step_batch(&self, actions: &Tensor) -> GpuStepResult;
    fn reset_batch(&self, mask: &Tensor) -> Tensor;
}

// All environment state on GPU
pub struct GpuCartPole {
    states: Tensor,      // [num_envs, 4]
    device: Device,
}

impl GpuEnv for GpuCartPole {
    fn step_batch(&self, actions: &Tensor) -> GpuStepResult {
        // Physics computed entirely on GPU
        let cos_theta = self.states.select(1, 2).cos();
        let sin_theta = self.states.select(1, 2).sin();
        // ... vectorized physics update
    }
}
```

---

## Profiling Commands

### CPU Profiling (Flamegraph)

```bash
# Install
cargo install flamegraph

# Profile
cargo flamegraph --release --features torch --bin puffer -- train cartpole --timesteps 100000

# View
open flamegraph.svg
```

### GPU Profiling (NVIDIA)

```bash
# nsys profiling
nsys profile --trace=cuda,nvtx cargo run --release --features torch -- train cartpole

# View
nsys-ui report.nsys-rep
```

### Memory Profiling

```bash
# Linux: heaptrack
heaptrack cargo run --release --features torch -- train cartpole --timesteps 10000
heaptrack_print heaptrack.*.gz | less

# Windows: Visual Studio Diagnostic Tools
# Or: valgrind --tool=massif (via WSL)
```

---

## Benchmark Suite

### Running Benchmarks

```bash
# All benchmarks
cargo bench --features torch

# Specific benchmark
cargo bench --features torch --bench vectorization

# Save baseline
cargo bench --features torch -- --save-baseline main

# Compare to baseline
cargo bench --features torch -- --baseline main
```

### Benchmark Categories

| Benchmark | File | Measures |
|:----------|:-----|:---------|
| `vecenv_step` | `benches/vectorization.rs` | Environment stepping throughput |
| `space_flatten` | `benches/spaces.rs` | Flattening/unflattening speed |
| `gae_compute` | `benches/training.rs` | Advantage computation |
| `policy_forward` | `benches/policy.rs` | Neural network latency |
| `buffer_add` | `benches/buffer.rs` | Experience buffer operations |

---

## Optimization Checklist

### Quick Wins (1-2 days each)
- [ ] Enable LTO in release builds
- [ ] Use `SmallVec` for fixed-size collections
- [ ] Pre-allocate `ExperienceBuffer` to max capacity
- [ ] Remove debug assertions in hot paths

### Medium Effort (1 week each)
- [ ] SIMD space flattening
- [ ] Batched GAE computation
- [ ] Memory-mapped observation buffers

### Major Effort (2-4 weeks each)
- [ ] Async trainer architecture
- [ ] GPU-native environments
- [ ] Multi-node distributed training

---

## Cargo.toml Optimization Settings

```toml
[profile.release]
lto = "fat"           # Link-time optimization
codegen-units = 1     # Single codegen unit for maximum optimization
opt-level = 3         # Maximum optimization
panic = "abort"       # No unwinding overhead

[profile.release.package."*"]
opt-level = 3

[profile.bench]
inherits = "release"
debug = true          # Keep symbols for profiling
```

---

## Hardware Recommendations

### Development
- 8+ CPU cores (Ryzen 7 / i7+)
- 32GB RAM
- NVIDIA GPU with 8GB+ VRAM

### Production Training
- 32+ CPU cores (EPYC / Xeon)
- 128GB RAM
- Multiple NVIDIA A100/H100 GPUs
- NVMe SSD for checkpoint I/O

---

*Document version: 1.0*
*Last updated: 2026-01-31*
