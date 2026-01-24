# Engineering Improvements: PufferLib Rust

## 1. Monomorphization of Vector Backends (Performance)
**Problem**: The current `Serial` and `Parallel` backends store environments as `Vec<Box<dyn PufferEnv>>`.
**Drawback**: This causes double indirection (Vec -> Box -> Struct) and vtable lookups on every `step()` call for every environment. It also destroys cache locality for environment states.
**Solution**: Refactor `Serial` and `Parallel` to be generic `Serial<E: PufferEnv>` and store `Vec<E>`.
**Benefit**: 
- Static dispatch (compiler inlines functions).
- Contiguous memory layout for environment states (better cache usage).
- No heap allocation for the `Box`.

## 2. Zero-Allocation `EnvInfo` (Performance/Memory)
**Problem**: `EnvInfo` uses `HashMap<String, f32>`.
**Drawback**: Allocates a new `HashMap` on the heap *every single step* for *every environment*. This is a massive performance killer in high-throughput RL.
**Solution**: 
- Use specific fields for common stats (`episode_return`, `episode_length`).
- Use `SmallVec` or fixed arrays for custom metrics if needed, or a simple `Vec<(&'static str, f32)>` which is cheaper than hashing.
- Ideally, return `EnvInfo` only when necessary, but the trait requires it.

## 3. Robust Error Handling (Reliability)
**Problem**: `vector/serial.rs` uses `expect` / `unwrap` on shape mismatches.
**Drawback**: Panics in production/training runs are unacceptable.
**Solution**: Propagate `PufferError` from vector backends.

## 4. Documentation & API Elegance
**Problem**: Missing docs on some standard methods.
**Solution**: standardizing doc comments.

## Phase Plan
1.  **Refactor `EnvInfo`**: Optimize data structure.
2.  **Refactor `PufferEnv` Trait**: Ensure it works with new `EnvInfo`.
3.  **Refactor `Serial` Backend**: meaningful monomorphization.
4.  **Refactor `Parallel` Backend**: meaningful monomorphization.
5.  **Update `pufferlib-envs`**: Adapt to new `EnvInfo` API.
6.  **Verify**: Run tests and demo.
