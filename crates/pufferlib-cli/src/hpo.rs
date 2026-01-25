//! CLI module for hyperparameter optimization.

use pufferlib::training::{TrainerConfig, hpo::{Study, SearchSpace, ParameterRange}};
use std::collections::HashMap;

/// Result of an HPO run
pub struct HpoResult {
    pub best_params: HashMap<String, f64>,
    pub best_value: f64,
}

/// Run an HPO study
pub fn run_hpo_study<F>(
    name: &str,
    space: SearchSpace,
    num_trials: usize,
    mut run_trial_func: F,
) -> HpoResult 
where 
    F: FnMut(usize, HashMap<String, f64>) -> f64
{
    let mut study = Study::new(name, space);

    for i in 0..num_trials {
        let params = study.suggest();
        let trial_id = study.create_trial(params.clone());
        
        tracing::info!(trial = trial_id, params = ?params, "Starting HPO trial");
        
        let value = run_trial_func(trial_id, params);
        
        study.complete_trial(trial_id, value);
        
        let best = study.best_trial().unwrap();
        tracing::info!(
            trial = trial_id, 
            value = value, 
            best_value = best.value.unwrap(),
            "Completed HPO trial"
        );
    }

    let best = study.best_trial().unwrap();
    HpoResult {
        best_params: best.params.clone(),
        best_value: best.value.unwrap(),
    }
}
