use crate::dapo::config::DapoConfig;
use crate::mappo::MultiAgentEnv;
use crate::policy::{DistributionSample, HasVarStore, Policy};
use tch::nn::OptimizerConfig;
use tch::{nn, Kind, Tensor};

/// Buffer to store rollout data for DAPO
pub(crate) struct DapoBuffer {
    pub(crate) obs: Vec<Tensor>,
    pub(crate) actions: Vec<Tensor>,
    pub(crate) log_probs: Vec<Tensor>,
    pub(crate) rewards: Vec<f32>,
    pub(crate) dones: Vec<bool>,
    /// Track actual context length for penalty
    pub(crate) lengths: Vec<usize>,
}

impl DapoBuffer {
    fn new() -> Self {
        Self {
            obs: Vec::new(),
            actions: Vec::new(),
            log_probs: Vec::new(),
            rewards: Vec::new(),
            dones: Vec::new(),
            lengths: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.obs.clear();
        self.actions.clear();
        self.log_probs.clear();
        self.rewards.clear();
        self.dones.clear();
        self.lengths.clear();
    }
}

pub struct DapoTrainer<P> {
    policy: P,
    ref_policy: Option<P>,
    optimizer: nn::Optimizer,
    config: DapoConfig,
    pub(crate) buffers: Vec<DapoBuffer>,
}

impl<P: Policy + HasVarStore> DapoTrainer<P> {
    pub fn new(mut policy: P, ref_policy: Option<P>, config: DapoConfig) -> Self {
        let optimizer = nn::Adam::default()
            .build(policy.var_store_mut(), config.learning_rate)
            .unwrap();

        Self {
            policy,
            ref_policy,
            optimizer,
            config,
            buffers: Vec::new(),
        }
    }

    pub fn collect_rollout<E: MultiAgentEnv>(&mut self, env: &mut E, num_steps: usize) {
        let mut obs = env.reset();
        let num_agents = obs.len();

        if self.buffers.len() != num_agents {
            self.buffers = (0..num_agents).map(|_| DapoBuffer::new()).collect();
        }

        for _ in 0..num_steps {
            let mut action_tensors = Vec::with_capacity(num_agents);

            for i in 0..num_agents {
                let agent_obs = &obs[i];
                let (dist, _, _) = self.policy.forward(agent_obs, &None);

                let action_sample = dist.sample();
                let log_prob_sample = dist.log_prob(&action_sample);

                let action = action_sample.as_torch();
                let log_prob = log_prob_sample.as_torch();

                self.buffers[i].obs.push(agent_obs.shallow_clone());
                self.buffers[i].actions.push(action.shallow_clone());
                self.buffers[i].log_probs.push(log_prob.detach());

                action_tensors.push(action.shallow_clone());
            }

            let result = env.step(&action_tensors);

            for (i, (reward, done)) in result.rewards.iter().zip(&result.dones).enumerate() {
                self.buffers[i].rewards.push(*reward);
                self.buffers[i].dones.push(*done);
                // In generic RL, "length" can just be incremented until done
                let current_len = self.buffers[i].lengths.last().cloned().unwrap_or(0);
                if *done {
                    self.buffers[i].lengths.push(0); // Start new episode length
                } else {
                    self.buffers[i].lengths.push(current_len + 1);
                }
            }

            obs = result.observations;
        }
    }

    pub fn update(&mut self) -> f64 {
        if self.buffers.is_empty() {
            return 0.0;
        }

        let mut all_returns = Vec::new();
        let mut all_obs = Vec::new();
        let mut all_actions = Vec::new();
        let mut all_old_log_probs = Vec::new();

        for buffer in &self.buffers {
            let mut returns = Vec::new();
            let mut R = 0.0;

            // Apply Length Penalty before return calculation
            let it = buffer
                .rewards
                .iter()
                .zip(&buffer.dones)
                .zip(&buffer.lengths)
                .rev();

            for ((&r, &d), &l) in it {
                if d {
                    R = 0.0;
                }

                // DAPO: Reward Shaping with length penalty for reasoning tasks
                let penalty = if l > self.config.target_max_length {
                    (l - self.config.target_max_length) as f32 * self.config.length_penalty_coef
                } else {
                    0.0
                };

                let shaped_reward = r - penalty;

                R = shaped_reward + (self.config.gamma as f32) * R;
                returns.push(R);
            }
            returns.reverse();

            all_returns.push(Tensor::from_slice(&returns));
            all_obs.push(Tensor::stack(&buffer.obs, 0));
            all_actions.push(Tensor::stack(&buffer.actions, 0));
            all_old_log_probs.push(Tensor::stack(&buffer.log_probs, 0));
        }

        // DAPO: Group advantage calculation with Dynamic Sampling
        let returns_t_n = Tensor::stack(&all_returns, 1); // [T, N]
        let group_size = self.config.group_size as i64;
        let total_samples = returns_t_n.size()[0] * returns_t_n.size()[1];

        // Ensure we have a multiple of group_size for reshaping
        let actual_total = (total_samples / group_size) * group_size;
        let num_groups = actual_total / group_size;

        if actual_total == 0 {
            return 0.0;
        }

        let rewards_flat = returns_t_n.flatten(0, 1).slice(0, 0, actual_total, 1);
        let rewards_grouped = rewards_flat.reshape(&[num_groups, group_size]); // [M, G]

        // Dynamic Sampling: identify informative groups
        let mask = if self.config.dynamic_sampling {
            let max_r = rewards_grouped.max_dim(1, true).0;
            let min_r = rewards_grouped.min_dim(1, true).0;
            (max_r - min_r).gt(1e-6).to_kind(Kind::Float)
        } else {
            Tensor::ones(&[num_groups, 1], (Kind::Float, rewards_grouped.device()))
        };

        let mean = rewards_grouped.mean_dim(Some(&[1i64][..]), true, Kind::Float);
        let std = rewards_grouped.std_dim(Some(&[1i64][..]), true, true);
        let advantages_grouped = (rewards_grouped - mean) / (std + 1e-5);

        // Apply mask to skip non-informative samples in gradient calculation
        let advantages = (advantages_grouped * mask).flatten(0, 1);

        let obs_flat = Tensor::stack(&all_obs, 1)
            .flatten(0, 1)
            .slice(0, 0, actual_total, 1);
        let actions_flat =
            Tensor::stack(&all_actions, 1)
                .flatten(0, 1)
                .slice(0, 0, actual_total, 1);
        let old_log_probs_flat =
            Tensor::stack(&all_old_log_probs, 1)
                .flatten(0, 1)
                .slice(0, 0, actual_total, 1);

        let mut total_loss = 0.0;
        for _ in 0..self.config.update_epochs {
            let (dist, _, _) = self.policy.forward(&obs_flat, &None);
            let action_sample = DistributionSample::Torch(actions_flat.shallow_clone());
            let new_log_probs = dist.log_prob(&action_sample).as_torch().shallow_clone();

            let ratio = (new_log_probs.shallow_clone() - &old_log_probs_flat).exp();
            let surr1 = &ratio * &advantages;

            // DAPO: Decoupled Clipping / Clip-Higher Strategy
            let clip_low = 1.0 - self.config.clip_coef_low;
            let clip_high = 1.0 + self.config.clip_coef_high;
            let surr2 = ratio.clamp(clip_low, clip_high) * &advantages;

            let ppo_loss = -surr1.min_other(&surr2).mean(Kind::Float);

            // KL Penalty if ref policy exists
            let kl_loss = if let Some(ref ref_pi) = self.ref_policy {
                let (ref_dist, _, _) = ref_pi.forward(&obs_flat, &None);
                let ref_log_probs = ref_dist.log_prob(&action_sample).as_torch().shallow_clone();
                (new_log_probs - ref_log_probs).mean(Kind::Float)
            } else {
                Tensor::zeros(&[], (Kind::Float, obs_flat.device()))
            };

            let loss = &ppo_loss + self.config.kl_coef * &kl_loss;

            self.optimizer.zero_grad();
            loss.backward();
            self.optimizer.step();

            total_loss += loss.double_value(&[]);
        }

        for buf in &mut self.buffers {
            buf.clear();
        }
        total_loss / self.config.update_epochs as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{MlpConfig, MlpPolicy};
    use tch::Device;

    #[test]
    #[cfg(feature = "torch")]
    fn test_dapo_update_dynamic_sampling() {
        let device = Device::Cpu;
        let obs_dim = 4;
        let action_dim = 2;
        let config = MlpConfig {
            hidden_size: 16,
            ..Default::default()
        };
        let policy = MlpPolicy::new(obs_dim as i64, action_dim as i64, false, config, device);

        let dapo_config = DapoConfig {
            group_size: 2,
            dynamic_sampling: true,
            ..Default::default()
        };

        let mut trainer = DapoTrainer::new(policy, None, dapo_config);

        // Setup 2 agents (1 group)
        trainer.buffers = (0..2).map(|_| DapoBuffer::new()).collect();

        // Agent 0 & 1: Identical rewards
        for i in 0..2 {
            trainer.buffers[i]
                .obs
                .push(Tensor::zeros([obs_dim as i64], (Kind::Float, device)));
            trainer.buffers[i]
                .actions
                .push(Tensor::zeros([1], (Kind::Float, device)));
            trainer.buffers[i]
                .log_probs
                .push(Tensor::zeros([1], (Kind::Float, device)));
            trainer.buffers[i].rewards.push(1.0);
            trainer.buffers[i].dones.push(false);
            trainer.buffers[i].lengths.push(1);
        }

        // Update should result in loss being 0.0 because of mask
        let loss = trainer.update();
        assert!(loss.abs() < 1e-6);
    }

    #[test]
    #[cfg(feature = "torch")]
    fn test_dapo_decoupled_clipping() {
        let device = Device::Cpu;
        let policy = MlpPolicy::new(
            2,
            2,
            false,
            MlpConfig {
                hidden_size: 8,
                ..Default::default()
            },
            device,
        );
        let config = DapoConfig {
            group_size: 2,
            clip_coef_low: 0.1,
            clip_coef_high: 0.9,
            update_epochs: 1,
            ..Default::default()
        };
        let mut trainer = DapoTrainer::new(policy, None, config);

        trainer.buffers = (0..2).map(|_| DapoBuffer::new()).collect();
        for i in 0..2 {
            trainer.buffers[i]
                .obs
                .push(Tensor::ones([2], (Kind::Float, device)));
            trainer.buffers[i]
                .actions
                .push(Tensor::zeros([1], (Kind::Float, device)));
            trainer.buffers[i].log_probs.push(Tensor::from(-1.0));
            trainer.buffers[i].rewards.push((i as f32) * 10.0);
            trainer.buffers[i].dones.push(false);
            trainer.buffers[i].lengths.push(1);
        }

        let loss = trainer.update();
        assert!(loss.abs() > 0.0 || loss == 0.0); // Should run without NaN
    }
}
