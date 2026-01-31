//! Metric logger traits and composites.

use crate::types::{Box, HashMap, String, Vec};

/// Trait for logging metrics to various backends.
pub trait MetricLogger: Send + Sync {
    /// Log a scalar value (e.g. reward, loss).
    fn log_scalar(&self, name: &str, value: f64, step: u64);

    /// Log a set of metrics collected in a dictionary/map.
    fn log_metrics(&self, metrics: &HashMap<String, f64>, step: u64);

    /// Close the logger and flush any pending writes.
    fn close(&self) {}
}

/// A logger that does nothing (default).
pub struct NoOpLogger;

impl MetricLogger for NoOpLogger {
    fn log_scalar(&self, _name: &str, _value: f64, _step: u64) {}
    fn log_metrics(&self, _metrics: &HashMap<String, f64>, _step: u64) {}
}

/// A composite logger that dispatches to multiple backends.
pub struct CompositeLogger {
    loggers: Vec<Box<dyn MetricLogger>>,
}

impl CompositeLogger {
    pub fn new(loggers: Vec<Box<dyn MetricLogger>>) -> Self {
        Self { loggers }
    }

    pub fn add(&mut self, logger: Box<dyn MetricLogger>) {
        self.loggers.push(logger);
    }
}

impl MetricLogger for CompositeLogger {
    fn log_scalar(&self, name: &str, value: f64, step: u64) {
        for logger in &self.loggers {
            logger.log_scalar(name, value, step);
        }
    }

    fn log_metrics(&self, metrics: &HashMap<String, f64>, step: u64) {
        for logger in &self.loggers {
            logger.log_metrics(metrics, step);
        }
    }

    fn close(&self) {
        for logger in &self.loggers {
            logger.close();
        }
    }
}
