//! Built-in environments for PufferLib.
//!
//! Provides simple environments for testing and benchmarking:
//! - `Bandit` - Multi-armed bandit
//! - `CartPole` - Classic control
//! - `Squared` - Grid navigation
//! - `Memory` - Sequence memorization

mod bandit;
mod cartpole;
mod hetero_mock;
mod memory;
mod mock_marl;
mod squared;

pub use bandit::Bandit;
pub use cartpole::CartPole;
pub use hetero_mock::HeteroMock;
pub use memory::Memory;
pub use mock_marl::MockMarl;
pub use squared::Squared;
