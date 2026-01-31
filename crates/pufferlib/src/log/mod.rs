//! Unified logging system.
//!
//! Provides:
//! - `MetricLogger` trait for composable backends
//! - `ConsoleLogger` for lightweight stdout logging
//! - `TensorBoardLogger` for visualization (optional)
//! - `CompositeLogger` for multi-backend logging

mod console;
mod logger;
#[cfg(feature = "tensorboard")]
mod tensorboard;

pub use console::ConsoleLogger;
pub use logger::{CompositeLogger, MetricLogger, NoOpLogger};
#[cfg(feature = "tensorboard")]
pub use tensorboard::TensorBoardLogger;
