//! Logging utilities for training.
use crate::training::TrainMetrics;
use serde_json::Value;
use std::collections::HashMap;

/// Trait for training loggers (W&B, TensorBoard, etc.)
pub trait Logger: Send + Sync {
    /// Initialize the logger with configuration
    fn init(
        &mut self,
        project: &str,
        run_name: &str,
        config: &HashMap<String, Value>,
    ) -> anyhow::Result<()>;

    /// Log metrics for a specific step
    fn log(&mut self, step: u64, metrics: &TrainMetrics) -> anyhow::Result<()>;

    /// Log extra custom metrics
    fn log_extra(&mut self, step: u64, extra: &HashMap<String, f64>) -> anyhow::Result<()>;

    /// Finalize logging for this run
    fn finalize(&mut self) -> anyhow::Result<()>;
}

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

/// Weights & Biases logger using Python bridge
#[cfg(feature = "python")]
pub struct WandbLogger {
    wandb: PyObject,
    run: Option<PyObject>,
}

#[cfg(feature = "python")]
impl WandbLogger {
    pub fn new() -> anyhow::Result<Self> {
        Python::with_gil(|py| {
            let wandb = py
                .import_bound("wandb")
                .map_err(|e| anyhow::anyhow!("Failed to import wandb: {}", e))?
                .to_object(py);
            Ok(Self { wandb, run: None })
        })
    }
}

#[cfg(feature = "python")]
impl Logger for WandbLogger {
    fn init(
        &mut self,
        project: &str,
        run_name: &str,
        config: &HashMap<String, Value>,
    ) -> anyhow::Result<()> {
        Python::with_gil(|py| {
            let wandb = self.wandb.bind(py);

            // Convert HashMap<String, Value> to Python dict
            let py_config = PyDict::new_bound(py);
            for (k, v) in config {
                let py_val = match v {
                    Value::Number(n) => n.as_f64().map(|f| f.to_object(py)),
                    Value::String(s) => Some(s.to_object(py)),
                    Value::Bool(b) => Some(b.to_object(py)),
                    _ => None, // Skip complex types for now
                };
                if let Some(val) = py_val {
                    py_config.set_item(k, val)?;
                }
            }

            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("project", project)?;
            kwargs.set_item("name", run_name)?;
            kwargs.set_item("config", py_config)?;

            let run = wandb.call_method("init", (), Some(&kwargs))?.to_object(py);

            self.run = Some(run);
            Ok(())
        })
    }

    fn log(&mut self, step: u64, metrics: &TrainMetrics) -> anyhow::Result<()> {
        let mut extra = HashMap::new();
        extra.insert("policy_loss".to_string(), metrics.policy_loss);
        extra.insert("value_loss".to_string(), metrics.value_loss);
        extra.insert("entropy".to_string(), metrics.entropy);
        extra.insert("kl".to_string(), metrics.kl);
        extra.insert("sps".to_string(), metrics.sps);
        self.log_extra(step, &extra)
    }

    fn log_extra(&mut self, step: u64, extra: &HashMap<String, f64>) -> anyhow::Result<()> {
        if let Some(ref run) = self.run {
            Python::with_gil(|py| {
                let py_metrics = PyDict::new_bound(py);
                for (k, v) in extra {
                    py_metrics.set_item(k, v)?;
                }
                py_metrics.set_item("global_step", step)?;

                run.bind(py).call_method("log", (py_metrics,), None)?;
                Ok(())
            })
        } else {
            Err(anyhow::anyhow!("WandbLogger not initialized"))
        }
    }

    fn finalize(&mut self) -> anyhow::Result<()> {
        if let Some(ref run) = self.run {
            Python::with_gil(|py| {
                run.bind(py).call_method0("finish")?;
                Ok(())
            })
        } else {
            Ok(())
        }
    }
}
