//! Multi-Agent PPO Trainer.

use super::buffer::AgentBuffer;
use super::config::MappoConfig;
use super::critic::CentralizedCritic;
use super::env::MultiAgentEnv;
use crate::policy::{HasVarStore, Policy};
use crate::training::{compute_gae, ppo_policy_loss, ppo_value_loss};
use tch::nn::OptimizerConfig;
use tch::{nn, Kind, Tensor};

/// MAPPO Trainer metrics
#[derive(Debug)]
pub struct MappoMetrics {
    pub policy_loss: f64,
    pub value_loss: f64,
    pub entropy: f64,
}

/// Multi-Agent PPO Trainer
pub struct MappoTrainer<P: Policy + HasVarStore> {
    /// Agent policies (shared or individual)
    policies: Vec<P>,
    /// Centralized critic
    critic: CentralizedCritic,
    /// Policy optimizer(s)
    policy_optimizers: Vec<nn::Optimizer>,
    /// Critic optimizer
    critic_optimizer: nn::Optimizer,
    /// Configuration
    config: MappoConfig,
    /// Per-agent experience buffers
    buffers: Vec<AgentBuffer>,
}

impl<P: Policy + HasVarStore> MappoTrainer<P> {
    pub fn new(mut policies: Vec<P>, mut critic: CentralizedCritic, config: MappoConfig) -> Self {
        let mut policy_optimizers = Vec::new();

        if config.share_policy {
            // Shared policy: 1 optimizer for the first policy (others are clones/synced if handled externally,
            // but here we likely assume policies[0] defines the shared weights)
            // Ideally if shared, we just have one policy instance, but to support Vec<P> generically:
            let opt = nn::Adam::default()
                .build(policies[0].var_store_mut(), config.learning_rate)
                .expect("Failed to create policy optimizer");
            policy_optimizers.push(opt);
        } else {
            // Independent policies
            for policy in &mut policies {
                let opt = nn::Adam::default()
                    .build(policy.var_store_mut(), config.learning_rate)
                    .expect("Failed to create policy optimizer");
                policy_optimizers.push(opt);
            }
        }

        let critic_optimizer = nn::Adam::default()
            .build(critic.var_store_mut(), config.learning_rate)
            .expect("Failed to create critic optimizer");

        let mut buffers = Vec::with_capacity(config.num_agents);
        for _ in 0..config.num_agents {
            buffers.push(AgentBuffer::new());
        }

        Self {
            policies,
            critic,
            policy_optimizers,
            critic_optimizer,
            config,
            buffers,
        }
    }

    /// Collect rollouts from all agents
    pub fn collect_rollout<E: MultiAgentEnv>(
        &mut self,
        env: &mut E,
        num_steps: usize,
    ) -> Vec<Tensor> {
        let mut obs = env.reset();
        let mut global_states = Vec::with_capacity(num_steps + 1);

        for _ in 0..num_steps {
            // Get global state for critic
            let global_state = env.get_global_state();
            global_states.push(global_state.shallow_clone());

            let global_value = self.critic.forward(&global_state);

            // Each agent selects action
            let mut action_tensors = Vec::with_capacity(self.config.num_agents);

            // If sharing policy, use policies[0], else policies[i]
            // We assume self.policies matches self.config.num_agents if not shared,
            // or we reuse policies[0] if shared.

            for i in 0..self.config.num_agents {
                let policy = if self.config.share_policy {
                    &self.policies[0]
                } else {
                    &self.policies[i]
                };

                let agent_obs = &obs[i];

                let state_none: Option<Vec<Tensor>> = None;
                let (dist, _, _) = policy.forward(agent_obs, &state_none);
                let action_sample = dist.sample();
                let log_prob_sample = dist.log_prob(&action_sample);

                let action = action_sample.as_torch();
                let log_prob = log_prob_sample.as_torch();

                // Store in agent's buffer
                self.buffers[i].add(
                    agent_obs.shallow_clone(),
                    action.reshape([-1]).detach(),
                    log_prob.reshape([-1]).detach(),
                    0.0, // reward filled after step
                    false,
                    global_value.reshape([-1]).detach(),
                );

                action_tensors.push(action.shallow_clone());
            }

            // Environment step
            let step_result = env.step(&action_tensors);

            // Update rewards and dones in buffers
            for (i, (reward, done)) in step_result
                .rewards
                .iter()
                .zip(&step_result.dones)
                .enumerate()
            {
                let buf = &mut self.buffers[i];
                if buf.len() > 0 {
                    let last_idx = buf.len() - 1;
                    buf.rewards[last_idx] = *reward;
                    buf.dones[last_idx] = *done;
                }
            }

            obs = step_result.observations;
        }

        // Get final global state for bootstrapping
        global_states.push(env.get_global_state());

        global_states
    }

    /// Perform MAPPO update
    pub fn update(&mut self, global_states: &[Tensor]) -> MappoMetrics {
        use crate::policy::DistributionSample;

        let mut total_policy_loss = 0.0;
        let mut total_value_loss = 0.0;
        let mut total_entropy = 0.0;

        // Compute advantages for all agents using global value
        let mut all_advantages = Vec::new();
        let mut all_returns = Vec::new();

        for (_i, buffer) in self.buffers.iter().enumerate() {
            let rewards = Tensor::from_slice(&buffer.rewards).unsqueeze(1);
            let values = Tensor::stack(&buffer.values, 0); // [Steps, 1]
            let dones_vec: Vec<f32> = buffer
                .dones
                .iter()
                .map(|&d| if d { 1.0 } else { 0.0 })
                .collect();
            let dones = Tensor::from_slice(&dones_vec).unsqueeze(1);

            // Last value from critic
            let last_global_state = &global_states[global_states.len() - 1];
            let last_value = self
                .critic
                .forward(last_global_state)
                .squeeze()
                .reshape([-1]); // [1]

            let advantages = compute_gae(
                &rewards,
                &values,
                &dones,
                &last_value,
                self.config.gamma,
                self.config.gae_lambda,
            );

            let returns = &advantages + &values;

            // Detach advantages and returns for PPO update
            let advantages = advantages.detach();
            let returns = returns.detach();

            all_advantages.push(advantages);
            all_returns.push(returns);
        }

        // PPO epochs
        for _epoch in 0..self.config.update_epochs {
            // Zero gradients once per epoch
            for opt in &mut self.policy_optimizers {
                opt.zero_grad();
            }
            self.critic_optimizer.zero_grad();

            let mut epoch_total_loss = None;

            for (i, buffer) in self.buffers.iter().enumerate() {
                let policy = if self.config.share_policy {
                    &self.policies[0]
                } else {
                    &self.policies[i]
                };

                // Stack buffer data
                let obs = Tensor::stack(&buffer.observations, 0);
                let actions = Tensor::stack(&buffer.actions, 0);
                let old_log_probs = Tensor::stack(&buffer.log_probs, 0).reshape([-1]);
                let advantages = &all_advantages[i].reshape([-1]);
                let returns = &all_returns[i];

                // Normalize advantages
                let advantages =
                    (advantages - advantages.mean(Kind::Float)) / (advantages.std(true) + 1e-8);

                // Forward pass
                let state_none: Option<Vec<Tensor>> = None;
                let (dist, _, _) = policy.forward(&obs, &state_none);

                // Wrap actions in DistributionSample for log_prob
                let actions_sample = DistributionSample::Torch(actions);
                let new_log_probs_sample = dist.log_prob(&actions_sample);
                let new_log_probs = new_log_probs_sample.as_torch().reshape([-1]);

                let entropy_sample = dist.entropy();
                let entropy = entropy_sample.as_torch();

                // Policy loss
                let policy_loss = ppo_policy_loss(
                    &advantages,
                    &new_log_probs,
                    &old_log_probs,
                    self.config.clip_coef,
                );

                // Value loss (use global states associated with this buffer's steps)
                let steps = buffer.len();
                let global_obs = Tensor::stack(&global_states[0..steps], 0);

                let new_values = self.critic.forward(&global_obs).reshape([-1, 1]);
                let old_values = Tensor::stack(&buffer.values, 0).reshape([-1, 1]);
                let returns_reshaped = returns.reshape([-1, 1]);

                let value_loss = ppo_value_loss(
                    &new_values,
                    &old_values,
                    &returns_reshaped,
                    self.config.clip_coef,
                );

                // Total loss for this agent
                let _agent_loss = &policy_loss + self.config.vf_coef * &value_loss
                    - self.config.ent_coef * entropy.mean(Kind::Float);

                total_policy_loss += policy_loss.double_value(&[]);
                total_value_loss += value_loss.double_value(&[]);
                total_entropy += entropy.mean(Kind::Float).double_value(&[]);

                // Accumulate split losses
                epoch_total_loss = Some(match epoch_total_loss {
                    None => (policy_loss, value_loss),
                    Some((p_acc, v_acc)) => (p_acc + policy_loss, v_acc + value_loss),
                });
            }

            // Perform backward pass once per epoch with accumulated loss
            if let Some((total_policy, total_value)) = epoch_total_loss {
                let combined_loss = total_policy + total_value * self.config.vf_coef;
                combined_loss.backward();

                // Step all optimizers
                for opt in &mut self.policy_optimizers {
                    opt.step();
                }
                self.critic_optimizer.step();
            }
        }

        // Clear buffers
        for buffer in &mut self.buffers {
            buffer.clear();
        }

        MappoMetrics {
            policy_loss: total_policy_loss
                / (self.config.update_epochs as f64 * self.config.num_agents as f64),
            value_loss: total_value_loss
                / (self.config.update_epochs as f64 * self.config.num_agents as f64),
            entropy: total_entropy
                / (self.config.update_epochs as f64 * self.config.num_agents as f64),
        }
    }
}
