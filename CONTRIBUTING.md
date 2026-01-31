# Contributing to PufferLib Rust

Thank you for your interest in contributing to PufferLib Rust!

## Getting Started

1.  **Fork the repository** and clone it locally.
2.  **Install dependencies**:
    -   Rust (stable)
    -   LibTorch (via `.\setup_libtorch.ps1` on Windows or following CI instructions for Linux)
    -   Protocol Buffers (optional, required for RPC)
3.  **Run tests**:
    ```bash
    cargo test --workspace --features torch
    ```

## Development Workflow

-   **Code Style**: We use standard `cargo fmt`. Please run it before submitting a PR.
-   **Linting**: We aim for zero Clippy warnings. Run `cargo clippy --workspace --features torch -- -D warnings`.
-   **Documentation**: If you add a new feature, please update `ARCHITECTURE.md` or provide a usage example in `README.md`.

## Project Structure

-   `crates/pufferlib`: Core logic (Spaces, Policies, Trainers).
-   `crates/pufferlib-envs`: Standard RL environments.
-   `crates/pufferlib-cli`: The `puffer` binary.
-   `crates/pufferlib-python`: PyO3 bindings.
-   `crates/pufferlib-bevy`: Bevy game engine integration.

## Centralized Versioning

The project uses a root `VERSIONS` file to manage LibTorch and CUDA versions across local scripts and GitHub Actions. Please update this file instead of hardcoding versions in CI or scripts.