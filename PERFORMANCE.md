# ⚡ Performance Strategy: PufferLib Rust

Rust's primary advantage in the RL ecosystem is throughput. This document defines how we measure and maintain that edge.

## 1. Benchmarking Targets
Our objective is to outperform Python-based [PufferLib](https://github.com/PufferAI/PufferLib) by **2-5x** in environment steps per second (SPS).

| Component | Target Optimization | Tooling |
| :--- | :--- | :--- |
| **Environment Step** | Zero-copy `ndarray` access | `cargo bench` |
| **Rollout Cycle** | Rayon work-stealing efficiency | `coz` (Causal Profiling) |
| **Neural Forward** | Batch latency (Torch/LibTorch) | `torch-profile` |
| **Space Flattening** | SIMD-accelerated recursion | `criterion` |

## 2. Profiling Workflow
To diagnose performance bottlenecks:

### 2.1 Macro-benchmarks (Wall-clock)
Use the included CLI to measure full-system throughput:
```bash
cargo run --release --bin puffer -- train cartpole --timesteps 1000000
# Monitor SPS in the TUI dashboard.
```

### 2.2 Micro-benchmarks
Located in `benches/`. Run via:
```bash
cargo bench
```

## 3. High-Fidelity Comparison
We maintain a suite of "Parity Environments" (CartPole, Bandit) that exist in both Python and Rust. Monthly audits will compare:
- **SPS (Steps Per Second)**: Raw simulation speed.
- **Latency (ms)**: Time from observation to action.
- **CPU/VRAM usage**: Memory efficiency during large-scale vectorization.
