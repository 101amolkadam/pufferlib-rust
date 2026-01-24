//! Identity environment with continuous action space for testing.

use ndarray::{ArrayD, IxDyn};
use pufferlib::env::{Action, EnvInfo, Observation, RawPufferEnv, RawStepResult};
use pufferlib::spaces::{Box as BoxSpace, DynSpace};

/// Environment where the observation is a random vector and reward is negative distance to action.
pub struct IdentityContinuous {
    target: Vec<f32>,
    size: usize,
}

impl IdentityContinuous {
    pub fn new(size: usize) -> Self {
        Self {
            target: vec![0.0; size],
            size,
        }
    }
}

impl Default for IdentityContinuous {
    fn default() -> Self {
        Self::new(4)
    }
}

impl RawPufferEnv for IdentityContinuous {
    fn observation_space(&self) -> DynSpace {
        DynSpace::Box(BoxSpace::uniform(&[self.size], -1.0, 1.0))
    }

    fn action_space(&self) -> DynSpace {
        DynSpace::Box(BoxSpace::uniform(&[self.size], -1.0, 1.0))
    }

    fn reset(&mut self, _seed: Option<u64>) -> (Observation, EnvInfo) {
        self.target = (0..self.size).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect();
        (
            Observation::Array(ArrayD::from_shape_vec(IxDyn(&[self.size]), self.target.clone()).unwrap()),
            EnvInfo::new(),
        )
    }

    fn step(&mut self, action: &Action) -> RawStepResult {
        let action_vec = match action {
            Action::Array(a) => a.as_slice().unwrap(),
            _ => panic!("Expected array action"),
        };

        let mut dist_sq = 0.0;
        for (t, a) in self.target.iter().zip(action_vec.iter()) {
            dist_sq += (t - a).powi(2);
        }

        let reward = -dist_sq;
        
        // New target
        self.target = (0..self.size).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect();
        
        RawStepResult {
            observation: Observation::Array(ArrayD::from_shape_vec(IxDyn(&[self.size]), self.target.clone()).unwrap()),
            reward,
            terminated: false,
            truncated: false,
            info: EnvInfo::new(),
        }
    }
}
