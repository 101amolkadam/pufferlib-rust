# PufferLib Rust Audit Report 2 (2026-01-31)

## 1. Executive Summary
The project is in a solid state but has several architectural limitations in its integrations (Bevy, WASM) and a significant amount of dead code/linting issues. The core RL logic is well-tested (50/50 tests passing), but some advanced features (Distributed, Dreamer) appear incomplete or unused in the current CLI.

## 2. Critical Issues
- **Dependency Conflict**: `burn-core` (0.13) had a transitive dependency on `tch` 0.15, conflicting with the required `tch` 0.23 for LibTorch 2.10.0.
  - *Status*: Partially resolved by disabling `burn-core` default features, but this may limit Burn's functionality.
- **Missing Build Dependency**: `pufferlib-rpc` requires `protoc` to be installed on the system, which causes build failures in environments without it.
  - *Recommendation*: Add a check in `build.rs` to skip proto generation or provide pre-generated bindings.

## 3. Medium Priority - Code Quality (Clippy Findings)
- **Dead Code**: Many structures and fields in `distributed.rs`, `exploration.rs`, `dreamer/`, and `trainer.rs` are never constructed or read.
- **API Inconsistency**: Several types implement `new()` but not `Default`, which is unidiomatic in Rust.
- **Casting Issues**: Unnecessary raw pointer casts in `vecenv.rs` and redundant `as f64` casts in `safe_ppo.rs`.
- **Naming Conventions**: `DAPO` and `GRPO` trainers use uppercase `R` for reward variables, violating Rust's snake_case convention.
- **Argument Bloat**: `AgentBuffer::add` takes 9 arguments.
  - *Recommendation*: Use a `Step` struct to group these arguments.

## 4. Technical Debt & Design Flaws
- **Bevy Integration**: `PufferBevyEnv` currently only supports a single agent (ID 0) and is hardcoded to use `PostUpdate`.
- **WASM Integration**: `PufferWasmEnv` is hardcoded to `CartPole`. It should be generic over `PufferEnv`.
- **Error Handling**: Many `.unwrap()` calls in `RemoteEnv` and `Win32SharedBuffer`. These should be converted to `Result` with proper error propagation.
- **Thread Stack Size**: `pufferlib-cli` hardcodes an 8MB stack size for the training thread. This should be configurable.

## 5. Security Considerations
- **Shared Memory**: `Win32SharedBuffer` uses named file mappings. While efficient, there is no validation of the buffer's contents before use, which could lead to memory safety issues if a malicious process modifies the shared memory.
- **Input Validation**: The CLI does not rigorously validate environment names before attempting to use them in match arms.

## 6. Documentation Audit
- **README.md**: Good high-level overview, but installation instructions for LibTorch were slightly outdated (fixed previously).
- **Architecture**: `ARCHITECTURE.md` is excellent but doesn't detail the `pufferlib-rpc` or `pufferlib-bevy` data flows in depth.
- **Missing**: Developer guide for adding new environments or custom reward models.

## 7. Next Steps
1.  **Refactor `AgentBuffer::add`** to use a struct.
2.  **Fix naming conventions** in trainers.
3.  **Implement `Default`** for loggers and toolkits.
4.  **Make `PufferWasmEnv` generic**.
5.  **Expand `PufferBevyEnv`** to support multi-agent setups.