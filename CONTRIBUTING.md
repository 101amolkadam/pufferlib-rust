# Contributing to PufferLib Rust

We love your input! We want to make contributing to PufferLib Rust as easy and transparent as possible.

## 🧪 Development Process

1. **Fork** the repo and create your branch from `main`.
2. **Update** documentation if you add new features.
3. **Check** your code with `cargo clippy` and `cargo fmt`.
4. **Test** your changes with `cargo test --workspace`.
5. **Issue** a pull request!

## 🌊 Porting Ocean Environments
"Ocean" is our suite of ultra-fast first-party environments. If you want to port a C environment from the original PufferLib:

1. **Native FFI**: Use `bindgen` to generate Rust bindings for the environment's C header.
2. **Trait Implementation**: Wrap the C state in a Rust struct and implement `PufferEnv`.
3. **Zero-Copy**: ensure the `step` function writes observations directly into the provided `ndarray`.

## 🛠️ Workflow Safety
- **No Force Push**: Never use `git push --force`. Always `pull` and merge.
- **Atomic Commits**: Small, single-purpose commits are preferred.

## 📜 Coding Standards

- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/).
- Use `anyhow` or `thiserror` for error handling.
- Document all public functions with doc comments.

## 🐞 Reporting Bugs

Use GitHub issues to report bugs. Please include:
- Your OS and Rust version.
- A minimal reproducible example.
- Expected vs. Actual behavior.
