//! Bayesian Hyperparameter Optimization (HPO) system.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rand::{Rng, distributions::{Distribution as RandDistribution, Uniform}};
use rand::seq::SliceRandom;

/// Types of hyperparameter ranges
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ParameterRange {
    /// Categorical choice (e.g. selection from a list of values)
    Categorical(Vec<String>),
    /// Uniform float range [min, max]
    Uniform(f64, f64),
    /// Log-uniform float range [min, max]
    LogUniform(f64, f64),
    /// Uniform integer range [min, max]
    IntUniform(i64, i64),
}

impl ParameterRange {
    /// Sample a value from the range
    pub fn sample(&self) -> f64 {
        let mut rng = rand::thread_rng();
        match self {
            ParameterRange::Categorical(choices) => {
                // Return index as float? Or need a way to return strings?
                // For simplified HPO, we usually map categories to integers internal to the optimizer.
                let idx = rng.gen_range(0..choices.len());
                idx as f64
            }
            ParameterRange::Uniform(min, max) => {
                let dist = Uniform::new(min, max);
                dist.sample(&mut rng)
            }
            ParameterRange::LogUniform(min, max) => {
                let log_min = min.ln();
                let log_max = max.ln();
                let dist = Uniform::new(log_min, log_max);
                dist.sample(&mut rng).exp()
            }
            ParameterRange::IntUniform(min, max) => {
                let dist = Uniform::new(min, max + 1);
                dist.sample(&mut rng) as f64
            }
        }
    }
}

/// A search space for hyperparameter optimization
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct SearchSpace {
    pub parameters: HashMap<String, ParameterRange>,
}

impl SearchSpace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, name: &str, range: ParameterRange) {
        self.parameters.insert(name.to_string(), range);
    }

    pub fn sample(&self) -> HashMap<String, f64> {
        let mut values = HashMap::new();
        for (name, range) in &self.parameters {
            values.insert(name.clone(), range.sample());
        }
        values
    }
}

/// A single trial in an HPO study
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Trial {
    pub id: usize,
    pub params: HashMap<String, f64>,
    pub value: Option<f64>,
    pub status: TrialStatus,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum TrialStatus {
    Running,
    Complete,
    Pruned,
    Fail,
}

/// A study containing multiple HPO trials
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Study {
    pub name: String,
    pub search_space: SearchSpace,
    pub trials: Vec<Trial>,
}

impl Study {
    pub fn new(name: &str, search_space: SearchSpace) -> Self {
        Self {
            name: name.to_string(),
            search_space,
            trials: Vec::new(),
        }
    }

    pub fn create_trial(&mut self, params: HashMap<String, f64>) -> usize {
        let id = self.trials.len();
        self.trials.push(Trial {
            id,
            params,
            value: None,
            status: TrialStatus::Running,
        });
        id
    }

    pub fn complete_trial(&mut self, trial_id: usize, value: f64) {
        if let Some(trial) = self.trials.get_mut(trial_id) {
            trial.value = Some(value);
            trial.status = TrialStatus::Complete;
        }
    }

    pub fn best_trial(&self) -> Option<&Trial> {
        self.trials.iter()
            .filter(|t| t.status == TrialStatus::Complete)
            .max_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Suggest new parameters using a simple Bayesian logic (TPE-inspired)
    pub fn suggest(&self) -> HashMap<String, f64> {
        // If we don't have enough trials, use random search
        if self.trials.len() < 10 {
            return self.search_space.sample();
        }

        // Simplified TPE:
        // 1. Split trials into "best" (top 25%) and "others".
        // 2. Pick a parameter set from "best" and add small noise.
        let mut complete_trials: Vec<&Trial> = self.trials.iter()
            .filter(|t| t.status == TrialStatus::Complete)
            .collect();
        
        if complete_trials.is_empty() {
            return self.search_space.sample();
        }

        complete_trials.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap_or(std::cmp::Ordering::Equal));
        
        let top_n = (complete_trials.len() as f64 * 0.25).max(1.0) as usize;
        let best_trials = &complete_trials[..top_n];

        let mut rng = rand::thread_rng();
        let selected_trial = best_trials.choose(&mut rng).unwrap();
        
        let mut suggested_params = HashMap::new();
        for (name, range) in &self.search_space.parameters {
            let current_val = *selected_trial.params.get(name).unwrap();
            
            // Add Gaussian noise (simplified Bayesian move)
            let val = match range {
                ParameterRange::Uniform(min, max) => {
                    let noise = rng.gen_range(-1.0..1.0) * (max - min) * 0.1;
                    (current_val + noise).clamp(*min, *max)
                }
                ParameterRange::LogUniform(min, max) => {
                    let log_val = current_val.ln();
                    let log_min = min.ln();
                    let log_max = max.ln();
                    let noise = rng.gen_range(-1.0..1.0) * (log_max - log_min) * 0.1;
                    (log_val + noise).clamp(log_min, log_max).exp()
                }
                ParameterRange::IntUniform(min, max) => {
                    let noise = rng.gen_range(-2..=2);
                    ((current_val as i64 + noise) as f64).clamp(*min as f64, *max as f64)
                }
                ParameterRange::Categorical(_) => {
                    // For categorical, either keep or pick random
                    if rng.gen_bool(0.8) { current_val } else { range.sample() }
                }
            };
            suggested_params.insert(name.clone(), val);
        }

        suggested_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hpo_basic() {
        let mut space = SearchSpace::new();
        space.add("lr", ParameterRange::LogUniform(1e-4, 1e-2));
        space.add("batch", ParameterRange::IntUniform(32, 256));

        let mut study = Study::new("test", space);
        for i in 0..15 {
            let params = study.suggest();
            let tid = study.create_trial(params);
            study.complete_trial(tid, i as f64);
        }

        let best = study.best_trial().unwrap();
        assert!(best.value == Some(14.0));
    }
}
