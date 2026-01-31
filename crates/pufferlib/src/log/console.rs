//! Console logging backend.

use super::MetricLogger;
use crate::types::{format, HashMap, String, Vec};

/// Logger that prints metrics to stdout via tracing.
pub struct ConsoleLogger;

impl Default for ConsoleLogger {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsoleLogger {
    pub fn new() -> Self {
        Self
    }
}

impl MetricLogger for ConsoleLogger {
    fn log_scalar(&self, name: &str, value: f64, step: u64) {
        tracing::info!("Step {}: {} = {:.4}", step, name, value);
    }

    fn log_metrics(&self, metrics: &HashMap<String, f64>, step: u64) {
        // Group output to avoid spamming lines
        let mut output = format!("Step {}: ", step);
        let mut sorted_keys: Vec<_> = metrics.keys().collect();
        sorted_keys.sort();

        for (i, key) in sorted_keys.iter().enumerate() {
            if i > 0 {
                output.push_str(", ");
            }
            output.push_str(&format!("{}={:.4}", key, metrics.get(*key).unwrap()));
        }

        tracing::info!("{}", output);
    }
}
