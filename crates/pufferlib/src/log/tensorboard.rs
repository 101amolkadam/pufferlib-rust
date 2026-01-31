//! TensorBoard logging backend.

use super::MetricLogger;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use tensorboard_rs::summary_writer::SummaryWriter;

/// Logger that writes to TensorBoard event files.
pub struct TensorBoardLogger {
    writer: Mutex<SummaryWriter>,
}

impl TensorBoardLogger {
    pub fn new(log_dir: impl AsRef<Path>) -> Self {
        let writer = SummaryWriter::new(log_dir.as_ref());
        Self {
            writer: Mutex::new(writer),
        }
    }
}

impl MetricLogger for TensorBoardLogger {
    fn log_scalar(&self, name: &str, value: f64, step: u64) {
        if let Ok(mut writer) = self.writer.lock() {
            writer.add_scalar(name, value as f32, step as usize);
            let _ = writer.flush();
        }
    }

    fn log_metrics(&self, metrics: &HashMap<String, f64>, step: u64) {
        if let Ok(mut writer) = self.writer.lock() {
            for (name, value) in metrics {
                writer.add_scalar(name, *value as f32, step as usize);
            }
            let _ = writer.flush();
        }
    }

    fn close(&self) {
        if let Ok(mut writer) = self.writer.lock() {
            let _ = writer.flush();
        }
    }
}
