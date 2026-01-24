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

/// Structured observation for complex spaces
#[derive(Clone, Debug)]
pub enum Observation {
    /// Primitive array
    Array(ArrayD<f32>),
    /// Dictionary of observations
    Dict(std::collections::HashMap<String, Observation>),
    /// Tuple of observations
    Tuple(Vec<Observation>),
}

/// Structured action for complex spaces
#[derive(Clone, Debug)]
pub enum Action {
    /// Primitive array
    Array(ArrayD<f32>),
    /// Dictionary of actions
    Dict(std::collections::HashMap<String, Action>),
    /// Tuple of actions
    Tuple(Vec<Action>),
}

/// Result from a raw environment step (unflattened)
#[derive(Clone, Debug)]
pub struct RawStepResult {
    /// Structured observation
    pub observation: Observation,
    /// Reward received
    pub reward: f32,
    /// Whether episode terminated
    pub terminated: bool,
    /// Whether episode truncated
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

/// Result from a multi-agent environment step
#[derive(Clone, Debug)]
pub struct MultiAgentStepResult {
    /// Observations for each agent
    pub observations: std::collections::HashMap<u32, ArrayD<f32>>,
    /// Rewards for each agent
    pub rewards: std::collections::HashMap<u32, f32>,
    /// Done flags for each agent
    pub terminated: std::collections::HashMap<u32, bool>,
    /// Truncated flags for each agent
    pub truncated: std::collections::HashMap<u32, bool>,
    /// Additional info
    pub info: EnvInfo,
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

    /// Optional: Take a multi-agent step
    ///
    /// # Arguments
    /// * `actions` - Map of agent ID to action
    fn multi_step(
        &mut self,
        _actions: &std::collections::HashMap<u32, ArrayD<f32>>,
    ) -> MultiAgentStepResult {
        unimplemented!("multi_step not implemented for this environment");
    }

    /// Optional: Render the environment
    fn render(&self) -> Option<String> {
        None
    }

    /// Optional: Close the environment and free resources
    fn close(&mut self) {}

    /// Get total number of agents (max population)
    fn num_agents(&self) -> usize {
        1
    }

    /// Get IDs of currently active agents
    fn active_agents(&self) -> Vec<u32> {
        vec![0]
    }

    /// Check if environment is done and needs reset
    fn is_done(&self) -> bool {
        false
    }
}

/// Trait for environments that return structured (unflattened) data.
/// 
/// The `EmulationLayer` wraps these to provide the standard `PufferEnv` interface.
pub trait RawPufferEnv: Send {
    /// Get the observation space
    fn observation_space(&self) -> DynSpace;

    /// Get the action space
    fn action_space(&self) -> DynSpace;

    /// Reset the environment
    fn reset(&mut self, seed: Option<u64>) -> (Observation, EnvInfo);

    /// Take a structured step
    fn step(&mut self, action: &Action) -> RawStepResult;

    /// Get total number of agents
    fn num_agents(&self) -> usize {
        1
    }

    /// Get IDs of currently active agents
    fn active_agents(&self) -> Vec<u32> {
        vec![0]
    }

    /// Take a multi-agent structured step
    fn multi_step(
        &mut self,
        _actions: &std::collections::HashMap<u32, Action>,
    ) -> std::collections::HashMap<u32, RawStepResult> {
        unimplemented!("multi_step not implemented");
    }

    /// Optional: Render the environment
    fn render(&self) -> Option<String> {
        None
    }

    /// Optional: Close the environment
    fn close(&mut self) {}

    /// Check if environment is done
    fn is_done(&self) -> bool {
        false
    }
}
