//! Curriculum learning systems.

use crate::training::self_play::PolicyPool;

/// Trait for defining training curricula
pub trait Curriculum: Send {
    /// Update the curriculum state based on current ELO ratings
    fn update(&mut self, pool: &PolicyPool);
    
    /// Get a parameter value for the current curriculum stage
    fn get_param(&self, key: &str) -> f32;
}

/// A simple curriculum that scales a parameter linearly with mean ELO
pub struct SimpleCurriculum {
    pub param_key: String,
    pub min_val: f32,
    pub max_val: f32,
    pub min_elo: f64,
    pub max_elo: f64,
    current_val: f32,
}

impl SimpleCurriculum {
    pub fn new(key: &str, min_val: f32, max_val: f32, min_elo: f64, max_elo: f64) -> Self {
        Self {
            param_key: key.to_string(),
            min_val,
            max_val,
            min_elo,
            max_elo,
            current_val: min_val,
        }
    }
}

impl Curriculum for SimpleCurriculum {
    fn update(&mut self, pool: &PolicyPool) {
        let policies = pool.all_policies();
        if policies.is_empty() { return; }
        
        let mean_elo: f64 = policies.iter().map(|p| p.rating).sum::<f64>() / policies.len() as f64;
        
        let t = ((mean_elo - self.min_elo) / (self.max_elo - self.min_elo))
            .clamp(0.0, 1.0) as f32;
            
        self.current_val = self.min_val + t * (self.max_val - self.min_val);
    }

    fn get_param(&self, _key: &str) -> f32 {
        self.current_val
    }
}
