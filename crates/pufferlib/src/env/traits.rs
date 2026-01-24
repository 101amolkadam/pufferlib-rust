//! Core environment trait definitions.

use crate::spaces::DynSpace;
use ndarray::ArrayD;

/// Information returned from environment steps
#[derive(Clone, Debug, Default)]
pub struct EnvInfo {
    /// Episode return (if done)
    pub episode_return: Option<f32>,
    /// Episode length (if done)
    pub episode_length: Option<f32>,
    /// Custom metrics (kept minimal for performance)
    pub extra: smallvec::SmallVec<[(&'static str, f32); 4]>,
}

impl EnvInfo {
    /// Create empty info
    pub fn new() -> Self {
        Self::default()
    }

    /// Add episode stats
    pub fn with_episode_stats(mut self, ret: f32, len: u32) -> Self {
        self.episode_return = Some(ret);
        self.episode_length = Some(len as f32);
        self
    }

    /// Add a custom metric (use rarely)
    pub fn with_extra(mut self, key: &'static str, value: f32) -> Self {
        self.extra.push((key, value));
        self
    }

    /// Get a value by key (including defaults)
    pub fn get(&self, key: &str) -> Option<f32> {
        match key {
            "episode_return" => self.episode_return,
            "episode_length" => self.episode_length,
            _ => self.extra.iter().find(|(k, _)| k == &key).map(|(_, v)| *v),
        }
    }
}

/// Result from a single environment step
#[derive(Clone, Debug)]
pub struct StepResult {
    /// Observation after the step
    pub observation: ArrayD<f32>,
    /// Reward received
    pub reward: f32,
    /// Whether episode terminated (goal reached, failure, etc.)
    pub terminated: bool,
    /// Whether episode truncated (time limit, etc.)
    pub truncated: bool,
    /// Additional info
    pub info: EnvInfo,
}

impl StepResult {
    /// Check if episode is done (terminated or truncated)
    pub fn done(&self) -> bool {
        self.terminated || self.truncated
    }
}

/// Core trait for PufferLib environments.
///
/// All environments must implement this trait to work with the training system.
///
/// # Example
///
/// ```rust,ignore
/// use pufferlib::env::PufferEnv;
/// use pufferlib::spaces::{Discrete, Box as BoxSpace};
///
/// struct MyEnv {
///     state: f32,
/// }
///
/// impl PufferEnv for MyEnv {
///     fn observation_space(&self) -> DynSpace {
///         DynSpace::Box(BoxSpace::uniform(&[1], -1.0, 1.0))
///     }
///
///     fn action_space(&self) -> DynSpace {
///         DynSpace::Discrete(Discrete::new(2))
///     }
///
///     fn reset(&mut self, seed: Option<u64>) -> (ArrayD<f32>, EnvInfo) {
///         self.state = 0.0;
///         (ArrayD::from_elem(IxDyn(&[1]), self.state), EnvInfo::new())
///     }
///
///     fn step(&mut self, action: &ArrayD<f32>) -> StepResult {
///         // ... implement step logic
///     }
/// }
/// ```
pub trait PufferEnv: Send {
    /// Get the observation space
    fn observation_space(&self) -> DynSpace;

    /// Get the action space
    fn action_space(&self) -> DynSpace;

    /// Reset the environment to initial state
    ///
    /// # Arguments
    /// * `seed` - Optional random seed for reproducibility
    ///
    /// # Returns
    /// Tuple of (initial observation, info dict)
    fn reset(&mut self, seed: Option<u64>) -> (ArrayD<f32>, EnvInfo);

    /// Take a single step in the environment
    ///
    /// # Arguments
    /// * `action` - Action to execute
    ///
    /// # Returns
    /// StepResult containing observation, reward, done flags, and info
    fn step(&mut self, action: &ArrayD<f32>) -> StepResult;

    /// Optional: Render the environment
    fn render(&self) -> Option<String> {
        None
    }

    /// Optional: Close the environment and free resources
    fn close(&mut self) {}

    /// Get the number of agents (default 1 for single-agent)
    fn num_agents(&self) -> usize {
        1
    }

    /// Check if environment is done and needs reset
    fn is_done(&self) -> bool {
        false
    }
}
