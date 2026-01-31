//! Multi-Agent Environment Trait.

use std::collections::HashMap;
use tch::Tensor;

/// Result of a single step in a multi-agent environment
pub struct MultiAgentStepResult {
    pub observations: Vec<Tensor>,
    pub rewards: Vec<f32>,
    pub dones: Vec<bool>,
    pub info: HashMap<String, String>,
    pub costs: Vec<f32>,
}

/// Trait for multi-agent environments compatible with MAPPO
pub trait MultiAgentEnv {
    /// Number of agents in the environment
    fn num_agents(&self) -> usize;

    /// Reset environment, returns observations for all agents
    fn reset(&mut self) -> Vec<Tensor>;

    /// Step environment with actions from all agents
    fn step(&mut self, actions: &[Tensor]) -> MultiAgentStepResult;

    /// Get global state (concatenated obs or privileged info)
    fn get_global_state(&self) -> Tensor;

    /// Get agent-specific observation
    fn get_observation(&self, agent_id: usize) -> Tensor;
}
