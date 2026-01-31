//! Experience buffer for storing rollout data.

use tch::{Device, Kind, Tensor};

/// Buffer for storing experience from environment rollouts
pub struct ExperienceBuffer {
    /// Observations
    pub observations: Tensor,
    /// Actions taken
    pub actions: Tensor,
    /// Log probabilities of actions
    pub log_probs: Tensor,
    /// Rewards received
    pub rewards: Tensor,
    /// Done flags (terminal or truncated)
    pub dones: Tensor,
    /// Value estimates
    pub values: Tensor,
    /// Advantages (computed after rollout)
    pub advantages: Tensor,
    /// Returns (advantages + values)
    /// Returns (advantages + values)
    pub returns: Tensor,
    /// Importance sampling weights (for V-trace)
    pub importance: Tensor,

    /// Costs received
    pub costs: Tensor,
    /// Cost value estimates
    pub cost_values: Tensor,
    /// Cost advantages
    pub cost_advantages: Tensor,
    /// Cost returns
    pub cost_returns: Tensor,

    /// Current position in buffer
    pos: usize,
    /// Buffer capacity
    capacity: usize,
    /// Number of environments
    num_envs: usize,
    /// Device
    device: Device,
}

impl ExperienceBuffer {
    /// Create a new experience buffer
    pub fn new(
        capacity: usize,
        num_envs: usize,
        obs_size: i64,
        action_size: i64,
        device: Device,
    ) -> Self {
        let total = (capacity * num_envs) as i64;

        Self {
            observations: Tensor::zeros([total, obs_size], (Kind::Float, device)),
            actions: Tensor::zeros([total, action_size], (Kind::Float, device)),
            log_probs: Tensor::zeros([total], (Kind::Float, device)),
            rewards: Tensor::zeros([total], (Kind::Float, device)),
            dones: Tensor::zeros([total], (Kind::Float, device)),
            values: Tensor::zeros([total], (Kind::Float, device)),
            advantages: Tensor::zeros([total], (Kind::Float, device)),
            returns: Tensor::zeros([total], (Kind::Float, device)),
            importance: Tensor::ones([total], (Kind::Float, device)),
            costs: Tensor::zeros([total], (Kind::Float, device)),
            cost_values: Tensor::zeros([total], (Kind::Float, device)),
            cost_advantages: Tensor::zeros([total], (Kind::Float, device)),
            cost_returns: Tensor::zeros([total], (Kind::Float, device)),
            pos: 0,
            capacity,
            num_envs,
            device,
        }
    }

    /// Add a batch of experience
    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &mut self,
        observations: &Tensor,
        actions: &Tensor,
        log_probs: &Tensor,
        rewards: &Tensor,
        dones: &Tensor,
        values: &Tensor,
        costs: &Tensor,
        cost_values: &Tensor,
    ) {
        let start = (self.pos * self.num_envs) as i64;
        let end = start + self.num_envs as i64;

        // Only copy if the source tensor is not already a view of our buffer memory
        // For Milestone 10, we usually write directly to the buffer in the worker loop.
        // If we did that, we can skip this copy.
        let mut obs_slice = self.observations.narrow(0, start, end - start);
        if !obs_slice.allclose(observations, 1e-8, 1e-8, false) {
            obs_slice.copy_(observations);
        }

        self.actions.narrow(0, start, end - start).copy_(actions);
        self.log_probs
            .narrow(0, start, end - start)
            .copy_(log_probs);
        self.rewards.narrow(0, start, end - start).copy_(rewards);
        self.dones.narrow(0, start, end - start).copy_(dones);
        self.values.narrow(0, start, end - start).copy_(values);
        self.costs.narrow(0, start, end - start).copy_(costs);
        self.cost_values
            .narrow(0, start, end - start)
            .copy_(cost_values);
        let _ = self.importance.narrow(0, start, end - start).fill_(1.0);

        self.pos += 1;
    }

    /// Reset buffer position
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.pos >= self.capacity
    }

    /// Get total number of samples
    pub fn len(&self) -> usize {
        self.pos * self.num_envs
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.pos == 0
    }

    /// Compute returns and advantages using GAE
    pub fn compute_returns_and_advantages(
        &mut self,
        last_value: &Tensor,
        gamma: f64,
        gae_lambda: f64,
    ) {
        let steps = self.pos;
        let mut last_gae = Tensor::zeros([self.num_envs as i64], (Kind::Float, self.device));
        let last_val = last_value.shallow_clone();

        for t in (0..steps).rev() {
            let start = (t * self.num_envs) as i64;
            let end = start + self.num_envs as i64;

            let next_values = if t == steps - 1 {
                last_val.shallow_clone()
            } else {
                let next_start = ((t + 1) * self.num_envs) as i64;
                self.values
                    .narrow(0, next_start, self.num_envs as i64)
                    .shallow_clone()
            };

            let rewards = self.rewards.narrow(0, start, end - start);
            let dones = self.dones.narrow(0, start, end - start);
            let values = self.values.narrow(0, start, end - start);

            let delta = &rewards + gamma * &next_values * (1.0 - &dones) - &values;
            last_gae = &delta + gamma * gae_lambda * (1.0 - &dones) * &last_gae;

            self.advantages
                .narrow(0, start, end - start)
                .copy_(&last_gae);
            self.returns
                .narrow(0, start, end - start)
                .copy_(&(&last_gae + &values));
        }
    }

    /// Compute cost returns and advantages using GAE
    pub fn compute_cost_advantages(
        &mut self,
        last_cost_value: &Tensor,
        gamma: f64,
        gae_lambda: f64,
    ) {
        let steps = self.pos;
        let mut last_gae = Tensor::zeros([self.num_envs as i64], (Kind::Float, self.device));
        let last_val = last_cost_value.shallow_clone();

        for t in (0..steps).rev() {
            let start = (t * self.num_envs) as i64;
            let end = start + self.num_envs as i64;

            let next_values = if t == steps - 1 {
                last_val.shallow_clone()
            } else {
                let next_start = ((t + 1) * self.num_envs) as i64;
                self.cost_values
                    .narrow(0, next_start, self.num_envs as i64)
                    .shallow_clone()
            };

            let costs = self.costs.narrow(0, start, end - start);
            let dones = self.dones.narrow(0, start, end - start);
            let values = self.cost_values.narrow(0, start, end - start);

            let delta = &costs + gamma * &next_values * (1.0 - &dones) - &values;
            last_gae = &delta + gamma * gae_lambda * (1.0 - &dones) * &last_gae;

            self.cost_advantages
                .narrow(0, start, end - start)
                .copy_(&last_gae);
            self.cost_returns
                .narrow(0, start, end - start)
                .copy_(&(&last_gae + &values));
        }
    }

    /// Compute returns and advantages using V-trace
    pub fn compute_vtrace(
        &mut self,
        last_value: &Tensor,
        gamma: f64,
        gae_lambda: f64,
        rho_clip: f64,
        c_clip: f64,
    ) {
        use super::ppo::compute_vtrace;

        let steps = self.pos;
        let num_envs = self.num_envs as i64;

        // Reshape flat tensors to [T, N]
        let rewards = self
            .rewards
            .narrow(0, 0, steps as i64 * num_envs)
            .reshape([steps as i64, num_envs]);
        let values = self
            .values
            .narrow(0, 0, steps as i64 * num_envs)
            .reshape([steps as i64, num_envs]);
        let dones = self
            .dones
            .narrow(0, 0, steps as i64 * num_envs)
            .reshape([steps as i64, num_envs]);
        let importance = self
            .importance
            .narrow(0, 0, steps as i64 * num_envs)
            .reshape([steps as i64, num_envs]);

        let advantages = compute_vtrace(
            &rewards,
            &values,
            &dones,
            &importance,
            last_value,
            gamma,
            gae_lambda,
            rho_clip,
            c_clip,
        );

        // Flatten back
        self.advantages
            .narrow(0, 0, steps as i64 * num_envs)
            .copy_(&advantages.flatten(0, -1));
        self.returns.narrow(0, 0, steps as i64 * num_envs).copy_(
            &(&self.advantages.narrow(0, 0, steps as i64 * num_envs)
                + &self.values.narrow(0, 0, steps as i64 * num_envs)),
        );
    }

    /// Get a minibatch of indices
    pub fn get_minibatch_indices(&self, batch_size: usize) -> Tensor {
        let total = self.len() as i64;
        Tensor::randperm(total, (Kind::Int64, self.device)).narrow(0, 0, batch_size as i64)
    }

    /// Get minibatch by indices
    pub fn get_minibatch(&self, indices: &Tensor) -> MiniBatch {
        MiniBatch {
            observations: self.observations.index_select(0, indices).detach(),
            actions: self.actions.index_select(0, indices).detach(),
            log_probs: self.log_probs.index_select(0, indices).detach(),
            advantages: self.advantages.index_select(0, indices).detach(),
            returns: self.returns.index_select(0, indices).detach(),
            values: self.values.index_select(0, indices).detach(),
            costs: self.costs.index_select(0, indices).detach(),
            cost_advantages: self.cost_advantages.index_select(0, indices).detach(),
            cost_values: self.cost_values.index_select(0, indices).detach(),
            cost_returns: self.cost_returns.index_select(0, indices).detach(),
        }
    }
}

/// A minibatch of experience for training
pub struct MiniBatch {
    pub observations: Tensor,
    pub actions: Tensor,
    pub log_probs: Tensor,
    pub advantages: Tensor,
    pub returns: Tensor,
    pub values: Tensor,
    pub costs: Tensor,
    pub cost_advantages: Tensor,
    pub cost_values: Tensor,
    pub cost_returns: Tensor,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_creation() {
        let buffer = ExperienceBuffer::new(128, 4, 10, 1, Device::Cpu);
        assert_eq!(buffer.len(), 0);
        assert!(!buffer.is_full());
    }

    #[test]
    fn test_buffer_add() {
        let mut buffer = ExperienceBuffer::new(4, 2, 4, 1, Device::Cpu);

        let obs = Tensor::randn([2, 4], (Kind::Float, Device::Cpu));
        let actions = Tensor::zeros([2, 1], (Kind::Float, Device::Cpu));
        let log_probs = Tensor::zeros([2], (Kind::Float, Device::Cpu));
        let rewards = Tensor::ones([2], (Kind::Float, Device::Cpu));
        let dones = Tensor::zeros([2], (Kind::Float, Device::Cpu));
        let values = Tensor::zeros([2], (Kind::Float, Device::Cpu));

        let zero_costs = Tensor::zeros([2], (Kind::Float, Device::Cpu));
        let zero_values = Tensor::zeros([2], (Kind::Float, Device::Cpu));
        buffer.add(
            &obs,
            &actions,
            &log_probs,
            &rewards,
            &dones,
            &values,
            &zero_costs,
            &zero_values,
        );
        assert_eq!(buffer.len(), 2);

        buffer.add(
            &obs,
            &actions,
            &log_probs,
            &rewards,
            &dones,
            &values,
            &zero_costs,
            &zero_values,
        );
        buffer.add(
            &obs,
            &actions,
            &log_probs,
            &rewards,
            &dones,
            &values,
            &zero_costs,
            &zero_values,
        );
        buffer.add(
            &obs,
            &actions,
            &log_probs,
            &rewards,
            &dones,
            &values,
            &zero_costs,
            &zero_values,
        );
        assert!(buffer.is_full());
    }
}
