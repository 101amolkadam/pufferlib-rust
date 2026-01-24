//! Multi-armed bandit environment.

use ndarray::{ArrayD, IxDyn};
use pufferlib::env::{EnvInfo, PufferEnv, StepResult};
use pufferlib::spaces::{Box as BoxSpace, Discrete, DynSpace};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Multi-armed bandit environment
///
/// The agent must learn which arm gives the highest reward.
/// Rewards are determined by a fixed random seed, so all
/// instances have the same optimal arm.
pub struct Bandit {
    /// Number of arms
    num_actions: usize,
    /// Reward scale
    reward_scale: f32,
    /// Reward noise stddev
    reward_noise: f32,
    /// Optimal arm index
    solution_idx: usize,
    /// RNG for noise
    rng: StdRng,
}

impl Bandit {
    /// Create a new bandit environment
    pub fn new(num_actions: usize) -> Self {
        Self::with_config(num_actions, 1.0, 0.0, 42)
    }

    /// Create with full configuration
    pub fn with_config(
        num_actions: usize,
        reward_scale: f32,
        reward_noise: f32,
        hard_fixed_seed: u64,
    ) -> Self {
        // Use fixed seed to determine solution
        let mut seed_rng = StdRng::seed_from_u64(hard_fixed_seed);
        let solution_idx = seed_rng.gen_range(0..num_actions);

        Self {
            num_actions,
            reward_scale,
            reward_noise,
            solution_idx,
            rng: StdRng::from_entropy(),
        }
    }
}

impl PufferEnv for Bandit {
    fn observation_space(&self) -> DynSpace {
        DynSpace::Box(BoxSpace::uniform(&[1], -1.0, 1.0))
    }

    fn action_space(&self) -> DynSpace {
        DynSpace::Discrete(Discrete::new(self.num_actions))
    }

    fn reset(&mut self, seed: Option<u64>) -> (ArrayD<f32>, EnvInfo) {
        if let Some(s) = seed {
            self.rng = StdRng::seed_from_u64(s);
        }

        let obs = ArrayD::from_elem(IxDyn(&[1]), 1.0);
        (obs, EnvInfo::new())
    }

    fn step(&mut self, action: &ArrayD<f32>) -> StepResult {
        let action_idx = action.iter().next().unwrap().round() as usize;
        assert!(action_idx < self.num_actions);

        let correct = action_idx == self.solution_idx;
        let mut reward = if correct { 1.0 } else { 0.0 };

        if self.reward_noise > 0.0 {
            let noise: f32 = self.rng.gen::<f32>() * 2.0 - 1.0;
            reward += noise * self.reward_noise;
        }

        reward *= self.reward_scale;

        let obs = ArrayD::from_elem(IxDyn(&[1]), 1.0);

        StepResult {
            observation: obs,
            reward,
            terminated: true, // Episode ends after one step
            truncated: false,
            info: EnvInfo::new().with_extra("score", if correct { 1.0 } else { 0.0 }),
        }
    }

    fn render(&self) -> Option<String> {
        Some(format!("Bandit: solution arm = {}", self.solution_idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bandit_creation() {
        let env = Bandit::new(4);
        assert_eq!(env.num_actions, 4);
    }

    #[test]
    fn test_bandit_step() {
        let mut env = Bandit::new(4);
        env.reset(Some(42));

        // Try all actions
        for i in 0..4 {
            let action = ArrayD::from_elem(IxDyn(&[1]), i as f32);
            let result = env.step(&action);
            assert!(result.terminated);
        }
    }

    #[test]
    fn test_bandit_seed_consistency() {
        let mut env1 = Bandit::new(10);
        let mut env2 = Bandit::new(10);

        let (obs1, _) = env1.reset(Some(123));
        let (obs2, _) = env2.reset(Some(123));
        assert_eq!(obs1, obs2);

        let action = ArrayD::from_elem(IxDyn(&[1]), 0.0);
        let res1 = env1.step(&action);
        let res2 = env2.step(&action);
        assert_eq!(res1.reward, res2.reward);
    }
}
