//! Generic bridge for Gymnasium-like environments.

use crate::env::{EnvInfo, PufferEnv, StepResult};
use crate::spaces::DynSpace;
use ndarray::ArrayD;

/// Standard interface for external RL environments
pub trait GymEnv: Send {
    /// Advance the environment by one step
    fn step(&mut self, action: &ArrayD<f32>) -> (ArrayD<f32>, f32, bool, bool, String);

    /// Reset the environment to initial state
    fn reset(&mut self, seed: Option<u64>) -> (ArrayD<f32>, String);

    /// Get the observation space
    fn observation_space(&self) -> DynSpace;

    /// Get the action space
    fn action_space(&self) -> DynSpace;
}

/// Wrapper to convert any GymEnv into a PufferEnv
pub struct PufferGymWrapper<T: GymEnv> {
    pub env: T,
}

impl<T: GymEnv> PufferGymWrapper<T> {
    pub fn new(env: T) -> Self {
        Self { env }
    }
}

impl<T: GymEnv> PufferEnv for PufferGymWrapper<T> {
    fn observation_space(&self) -> DynSpace {
        self.env.observation_space()
    }

    fn action_space(&self) -> DynSpace {
        self.env.action_space()
    }

    fn reset(&mut self, seed: Option<u64>) -> (ArrayD<f32>, EnvInfo) {
        let (obs, _info_str) = self.env.reset(seed);
        (obs, EnvInfo::default())
    }

    fn step(&mut self, action: &ArrayD<f32>) -> StepResult {
        let (obs, reward, terminated, truncated, _info_str) = self.env.step(action);
        StepResult {
            observation: obs,
            reward,
            terminated,
            truncated,
            info: EnvInfo::default(),
        }
    }
}
