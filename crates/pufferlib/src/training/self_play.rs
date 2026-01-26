//! Self-play system for training against historical versions of the policy.

use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::path::PathBuf;

/// Entry in the policy pool
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct PolicyRecord {
    pub id: String,
    pub path: PathBuf,
    pub rating: f64,
    pub matches: usize,
}

/// A pool of historical policies for self-play
pub struct PolicyPool {
    policies: HashMap<String, PolicyRecord>,
    active_ids: Vec<String>,
}

impl PolicyPool {
    /// Create a new empty policy pool
    pub fn new() -> Self {
        Self {
            policies: HashMap::new(),
            active_ids: Vec::new(),
        }
    }

    /// Add a policy to the pool
    pub fn add_policy(&mut self, id: String, path: PathBuf, rating: f64) {
        let record = PolicyRecord {
            id: id.clone(),
            path,
            rating,
            matches: 0,
        };
        self.policies.insert(id.clone(), record);
        self.active_ids.push(id);
    }

    /// Sample a policy from the pool for matching
    pub fn sample_policy(&self) -> Option<&PolicyRecord> {
        let mut rng = rand::thread_rng();
        self.active_ids
            .choose(&mut rng)
            .and_then(|id| self.policies.get(id))
    }

    /// Update ELO ratings for a match
    #[allow(dead_code)]
    pub fn update_ratings(&mut self, winner_id: &str, loser_id: &str, draw: bool) {
        let k_factor = 32.0;

        let (r_w, r_l) = match (self.policies.get(winner_id), self.policies.get(loser_id)) {
            (Some(w), Some(l)) => (w.rating, l.rating),
            _ => return,
        };

        let expected_w = 1.0 / (1.0 + 10.0f64.powf((r_l - r_w) / 400.0));
        let expected_l = 1.0 / (1.0 + 10.0f64.powf((r_w - r_l) / 400.0));

        let actual_w = if draw { 0.5 } else { 1.0 };
        let actual_l = if draw { 0.5 } else { 0.0 };

        if let Some(w) = self.policies.get_mut(winner_id) {
            w.rating += k_factor * (actual_w - expected_w);
            w.matches += 1;
        }

        if let Some(l) = self.policies.get_mut(loser_id) {
            l.rating += k_factor * (actual_l - expected_l);
            l.matches += 1;
        }
    }

    /// Get a policy record by ID
    #[allow(dead_code)]
    pub fn get_policy(&self, id: &str) -> Option<&PolicyRecord> {
        self.policies.get(id)
    }

    /// Get all policy records
    pub fn all_policies(&self) -> Vec<&PolicyRecord> {
        self.policies.values().collect()
    }
}

/// Helper to update ELO between two ratings
#[allow(dead_code)]
pub fn compute_elo_update(rating_a: f64, rating_b: f64, score_a: f64, k: f64) -> f64 {
    let expected_a = 1.0 / (1.0 + 10.0f64.powf((rating_b - rating_a) / 400.0));
    rating_a + k * (score_a - expected_a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elo_logic() {
        let mut pool = PolicyPool::new();
        pool.add_policy("agent1".to_string(), PathBuf::from("p1"), 1000.0);
        pool.add_policy("agent2".to_string(), PathBuf::from("p2"), 1000.0);

        pool.update_ratings("agent1", "agent2", false);

        let p1 = pool.get_policy("agent1").unwrap();
        let p2 = pool.get_policy("agent2").unwrap();

        assert!(p1.rating > 1000.0);
        assert!(p2.rating < 1000.0);
        assert!(p1.matches == 1);
        assert!(p2.matches == 1);
    }
}
