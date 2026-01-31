//! Multi-Agent PPO implementation.
//!
//! Supports cooperative and competitive multi-agent scenarios using
//! Centralized Training, Decentralized Execution (CTDE).

#[cfg(feature = "torch")]
mod buffer;
#[cfg(feature = "torch")]
mod config;
#[cfg(feature = "torch")]
mod critic;
#[cfg(feature = "torch")]
mod env;
#[cfg(feature = "torch")]
mod trainer;

#[cfg(feature = "torch")]
pub use buffer::AgentBuffer;
#[cfg(feature = "torch")]
pub use config::MappoConfig;
#[cfg(feature = "torch")]
pub use critic::CentralizedCritic;
#[cfg(feature = "torch")]
pub use env::{MultiAgentEnv, MultiAgentStepResult};
#[cfg(feature = "torch")]
pub use trainer::{MappoMetrics, MappoTrainer};

#[cfg(test)]
#[cfg(feature = "torch")]
mod tests {
    use super::*;
    use crate::policy::{Distribution, HasVarStore, Policy};
    use std::collections::HashMap;
    use tch::{nn, Device, Kind, Tensor};

    struct MockMultiAgentEnv {
        num_agents: usize,
        obs_dim: i64,
        global_state_dim: i64,
        device: Device,
        step_count: usize,
    }

    impl MockMultiAgentEnv {
        fn new(num_agents: usize, obs_dim: i64, global_state_dim: i64) -> Self {
            Self {
                num_agents,
                obs_dim,
                global_state_dim,
                device: Device::Cpu,
                step_count: 0,
            }
        }
    }

    impl MultiAgentEnv for MockMultiAgentEnv {
        fn reset(&mut self) -> Vec<Tensor> {
            println!("Debug - Env Resetting");
            (0..self.num_agents)
                .map(|_| Tensor::randn(&[self.obs_dim], (Kind::Float, self.device)))
                .collect()
        }

        fn step(&mut self, actions: &[Tensor]) -> MultiAgentStepResult {
            println!("Debug - Env Stepped with {} actions", actions.len());
            self.step_count += 1;
            let obs = self.reset();
            let rewards = vec![1.0; self.num_agents];
            let dones = vec![self.step_count % 10 == 0; self.num_agents];

            MultiAgentStepResult {
                observations: obs,
                rewards,
                dones,
                info: HashMap::new(),
                costs: vec![0.0; self.num_agents],
            }
        }

        fn get_global_state(&self) -> Tensor {
            Tensor::randn(&[self.global_state_dim], (Kind::Float, self.device))
        }

        fn num_agents(&self) -> usize {
            self.num_agents
        }

        fn get_observation(&self, _agent_id: usize) -> Tensor {
            Tensor::randn(&[self.obs_dim], (Kind::Float, self.device))
        }
    }

    struct MockPolicy {
        var_store: nn::VarStore,
        linear: nn::Linear,
    }

    impl MockPolicy {
        fn new(obs_dim: i64, action_dim: i64) -> Self {
            let var_store = nn::VarStore::new(Device::Cpu);
            let linear = nn::linear(var_store.root(), obs_dim, action_dim, Default::default());
            Self { var_store, linear }
        }
    }

    impl HasVarStore for MockPolicy {
        fn var_store(&self) -> &nn::VarStore {
            &self.var_store
        }
        fn var_store_mut(&mut self) -> &mut nn::VarStore {
            &mut self.var_store
        }
    }

    impl Policy for MockPolicy {
        fn forward(
            &self,
            obs: &Tensor,
            _state: &Option<Vec<Tensor>>,
        ) -> (Distribution, Tensor, Option<Vec<Tensor>>) {
            let logits = obs.apply(&self.linear);
            let dist = Distribution::Categorical {
                logits: logits.shallow_clone(),
            };
            let size = if obs.dim() > 1 { obs.size()[0] } else { 1 };
            let value = Tensor::zeros([size, 1], (Kind::Float, Device::Cpu));
            (dist, value, None)
        }

        fn initial_state(&self, _batch_size: i64) -> Option<Vec<Tensor>> {
            None
        }
    }

    #[test]
    fn test_mappo_config_defaults() {
        let config = MappoConfig::default();
        assert_eq!(config.num_agents, 2);
        assert!(config.share_policy);
    }

    #[test]
    fn test_mappo_training_loop() {
        let num_agents = 2;
        let obs_dim = 4;
        let global_state_dim = 8;
        let action_dim = 3;

        let config = MappoConfig {
            num_agents,
            obs_dim,
            action_dim,
            global_state_dim,
            share_policy: true,
            use_global_state: true,
            clip_coef: 0.2,
            vf_coef: 0.5,
            ent_coef: 0.01,
            gamma: 0.99,
            gae_lambda: 0.95,
            update_epochs: 2,
            minibatch_size: None,
            learning_rate: 1e-3,
            max_grad_norm: 0.5,
        };

        let mut env = MockMultiAgentEnv::new(num_agents, obs_dim, global_state_dim);
        let policy = MockPolicy::new(obs_dim, action_dim);
        let critic = CentralizedCritic::new(global_state_dim, 64, Device::Cpu);

        let mut trainer = MappoTrainer::new(vec![policy], critic, config);

        // Collect rollout
        let global_states = trainer.collect_rollout(&mut env, 20);
        assert_eq!(global_states.len(), 21); // 20 steps + bootstrap

        // Update
        let metrics = trainer.update(&global_states);

        // Check metrics
        // We just ensure it ran without panicking and returned something
        println!("Policy Loss: {}", metrics.policy_loss);
    }
}
