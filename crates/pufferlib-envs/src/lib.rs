//! Built-in environments for PufferLib.
//!
//! Provides simple environments for testing and benchmarking:
//! - `Bandit` - Multi-armed bandit
//! - `CartPole` - Classic control
//! - `Squared` - Grid navigation
//! - `Memory` - Sequence memorization

mod bandit;
mod cartpole;
mod squared;
mod memory;

pub use bandit::Bandit;
pub use cartpole::CartPole;
pub use squared::Squared;
pub use memory::Memory;
