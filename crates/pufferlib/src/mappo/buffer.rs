//! Per-agent experience buffer.

use tch::Tensor;

/// Buffer for a single agent's experience
#[derive(Default)]
pub struct AgentBuffer {
    pub observations: Vec<Tensor>,
    pub actions: Vec<Tensor>,
    pub log_probs: Vec<Tensor>,
    pub rewards: Vec<f32>,
    pub dones: Vec<bool>,
    pub values: Vec<Tensor>,
}

impl AgentBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(
        &mut self,
        obs: Tensor,
        action: Tensor,
        log_prob: Tensor,
        reward: f32,
        done: bool,
        value: Tensor,
    ) {
        self.observations.push(obs);
        self.actions.push(action);
        self.log_probs.push(log_prob);
        self.rewards.push(reward);
        self.dones.push(done);
        self.values.push(value);
    }

    pub fn clear(&mut self) {
        self.observations.clear();
        self.actions.clear();
        self.log_probs.clear();
        self.rewards.clear();
        self.dones.clear();
        self.values.clear();
    }

    pub fn len(&self) -> usize {
        self.observations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }
}
