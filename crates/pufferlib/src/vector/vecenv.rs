//! Vectorized environment abstraction.

use crate::env::EnvInfo;
use crate::spaces::DynSpace;
use crate::types::{vec, Vec};
use ndarray::Array2;
#[cfg(feature = "torch")]
use tch::{Device, Kind, Tensor};

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

/// Observation batch that can reside on CPU or GPU
#[derive(Debug)]
pub enum ObservationBatch {
    /// Standard CPU array
    Cpu(Array2<f32>),
    /// LibTorch tensor (possibly on GPU)
    #[cfg(feature = "torch")]
    Torch(tch::Tensor),
    /// Candle tensor (possibly on GPU)
    #[cfg(feature = "candle")]
    Candle(candle_core::Tensor),
}

impl Clone for ObservationBatch {
    fn clone(&self) -> Self {
        match self {
            Self::Cpu(a) => Self::Cpu(a.clone()),
            #[cfg(feature = "torch")]
            Self::Torch(t) => Self::Torch(t.shallow_clone()),
            #[cfg(feature = "candle")]
            Self::Candle(t) => Self::Candle(t.clone()),
        }
    }
}

impl ObservationBatch {
    pub fn num_envs(&self) -> usize {
        match self {
            Self::Cpu(a) => a.shape()[0],
            #[cfg(feature = "torch")]
            Self::Torch(t) => t.size()[0] as usize,
            #[cfg(feature = "candle")]
            Self::Candle(t) => t.dims()[0],
        }
    }

    /// Create from shared memory (Zero-Copy View if Torch)
    pub fn from_shared(buffer: &dyn super::SharedBuffer, shape: &[i64]) -> Self {
        #[cfg(feature = "torch")]
        {
            let tensor = unsafe {
                Tensor::f_from_blob(
                    buffer.as_ptr() as *mut f32 as *mut _,
                    shape,
                    &[],
                    Kind::Float,
                    Device::Cpu,
                )
                .expect("Failed to create tensor from blob")
            };
            Self::Torch(tensor)
        }
        #[cfg(not(feature = "torch"))]
        {
            let len = buffer.len();
            let mut data = vec![0.0f32; len];
            unsafe {
                core::ptr::copy_nonoverlapping(buffer.as_ptr(), data.as_mut_ptr(), len);
            }
            let rows = shape[0] as usize;
            let cols = shape[1..].iter().product::<i64>() as usize;
            let array = Array2::from_shape_vec((rows, cols), data).unwrap();
            Self::Cpu(array)
        }
    }

    /// Deep copy for buffer storage
    pub fn to_owned(&self) -> Self {
        match self {
            Self::Cpu(a) => Self::Cpu(a.clone()),
            #[cfg(feature = "torch")]
            Self::Torch(t) => Self::Torch(t.shallow_clone().copy()),
            #[cfg(feature = "candle")]
            Self::Candle(t) => Self::Candle(t.clone()),
        }
    }
}

/// Result from stepping all environments
#[derive(Clone, Debug)]
pub struct VecEnvResult {
    /// Observations for all environments
    pub observations: ObservationBatch,
    /// Rewards for all environments
    pub rewards: Vec<f32>,
    /// Terminated flags
    pub terminated: Vec<bool>,
    /// Truncated flags
    pub truncated: Vec<bool>,
    /// Info dictionaries
    pub infos: Vec<EnvInfo>,
    /// Costs for all environments
    pub costs: Vec<f32>,
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
    fn reset(&mut self, seed: Option<u64>) -> (ObservationBatch, Vec<EnvInfo>);

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
    pub fn reset(&mut self, seed: Option<u64>) -> (ObservationBatch, Vec<EnvInfo>) {
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

    fn reset(&mut self, seed: Option<u64>) -> (ObservationBatch, Vec<EnvInfo>) {
        self.backend.reset(seed)
    }

    fn step(&mut self, actions: &Array2<f32>) -> VecEnvResult {
        self.backend.step(actions)
    }

    fn close(&mut self) {
        self.backend.close()
    }
}
