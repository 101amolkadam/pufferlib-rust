//! Vectorized environment abstraction.

use crate::env::EnvInfo;
use crate::spaces::DynSpace;
use ndarray::Array2;

/// Configuration for vectorized environments
#[derive(Clone, Debug)]
pub struct VecEnvConfig {
    /// Number of environments
    pub num_envs: usize,
    /// Random seed base
    pub seed: u64,
}

impl Default for VecEnvConfig {
    fn default() -> Self {
        Self {
            num_envs: 1,
            seed: 42,
        }
    }
}

impl VecEnvConfig {
    /// Create a new config with specified number of environments
    pub fn new(num_envs: usize) -> Self {
        Self {
            num_envs,
            ..Default::default()
        }
    }

    /// Set the random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Result from stepping all environments
#[derive(Clone, Debug)]
pub struct VecEnvResult {
    /// Observations for all environments (num_envs, *obs_shape)
    pub observations: Array2<f32>,
    /// Rewards for all environments
    pub rewards: Vec<f32>,
    /// Terminated flags
    pub terminated: Vec<bool>,
    /// Truncated flags
    pub truncated: Vec<bool>,
    /// Info dictionaries
    pub infos: Vec<EnvInfo>,
}

impl VecEnvResult {
    /// Check which environments are done
    pub fn dones(&self) -> Vec<bool> {
        self.terminated
            .iter()
            .zip(self.truncated.iter())
            .map(|(&t, &tr)| t || tr)
            .collect()
    }
}

/// Trait for vectorized environment backends
pub trait VecEnvBackend: Send {
    /// Get the observation space (single env)
    fn observation_space(&self) -> DynSpace;

    /// Get the action space (single env)
    fn action_space(&self) -> DynSpace;

    /// Get the number of environments
    fn num_envs(&self) -> usize;

    /// Reset all environments
    fn reset(&mut self, seed: Option<u64>) -> (Array2<f32>, Vec<EnvInfo>);

    /// Step all environments with given actions
    fn step(&mut self, actions: &Array2<f32>) -> VecEnvResult;

    /// Close all environments
    fn close(&mut self);
}

/// Main vectorized environment struct
pub struct VecEnv<B: VecEnvBackend> {
    backend: B,
}

impl<B: VecEnvBackend> VecEnv<B> {
    /// Create from a backend
    pub fn from_backend(backend: B) -> Self {
        Self { backend }
    }

    /// Get observation space
    pub fn observation_space(&self) -> DynSpace {
        self.backend.observation_space()
    }

    /// Get action space
    pub fn action_space(&self) -> DynSpace {
        self.backend.action_space()
    }

    /// Get number of environments
    pub fn num_envs(&self) -> usize {
        self.backend.num_envs()
    }

    /// Reset all environments
    pub fn reset(&mut self, seed: Option<u64>) -> (Array2<f32>, Vec<EnvInfo>) {
        self.backend.reset(seed)
    }

    /// Step all environments
    pub fn step(&mut self, actions: &Array2<f32>) -> VecEnvResult {
        self.backend.step(actions)
    }

    /// Close all environments
    pub fn close(&mut self) {
        self.backend.close()
    }
}

impl<B: VecEnvBackend> VecEnvBackend for VecEnv<B> {
    fn observation_space(&self) -> DynSpace {
        self.backend.observation_space()
    }

    fn action_space(&self) -> DynSpace {
        self.backend.action_space()
    }

    fn num_envs(&self) -> usize {
        self.backend.num_envs()
    }

    fn reset(&mut self, seed: Option<u64>) -> (Array2<f32>, Vec<EnvInfo>) {
        self.backend.reset(seed)
    }

    fn step(&mut self, actions: &Array2<f32>) -> VecEnvResult {
        self.backend.step(actions)
    }

    fn close(&mut self) {
        self.backend.close()
    }
}
