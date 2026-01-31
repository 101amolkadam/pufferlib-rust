//! Safe RL (Constrained PPO) trainer.

use crate::policy::{DistributionSample, HasVarStore, SafePolicy};
use crate::spaces::Space;
use crate::training::buffer::ExperienceBuffer;
use crate::training::config::{ConstrainedPpoConfig, TrainerConfig};
use crate::training::ppo::{ppo_policy_loss, ppo_value_loss};
use crate::vector::{ObservationBatch, VecEnvBackend};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray;
use std::time::Instant;
#[cfg(feature = "torch")]
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

/// Trainer for Constrained PPO (Safe RL)
pub struct SafeTrainer<P: SafePolicy + HasVarStore, V: VecEnvBackend> {
    config: TrainerConfig,
    safe_config: ConstrainedPpoConfig,
    vecenv: V,
    policy: P,
    optimizer: nn::Optimizer,
    buffer: ExperienceBuffer,
    global_step: u64,
    epoch: u64,
    start_time: Instant,
    num_envs: usize,
    obs_size: i64,
    action_size: i64,
    state: Option<Vec<Tensor>>,
    progress: Option<ProgressBar>,

    /// Lagrangian multiplier (lambda)
    lagrangian: Tensor,
    /// Lagrangian optimizer
    lagrangian_optimizer: nn::Optimizer,

    pub last_loss: f64,
    pub last_cost: f64,
    pub mean_reward: f64,
}

impl<P: SafePolicy + HasVarStore, V: VecEnvBackend> SafeTrainer<P, V> {
    pub fn new(
        vecenv: V,
        mut policy: P,
        config: TrainerConfig,
        safe_config: ConstrainedPpoConfig,
        device: Device,
    ) -> Self {
        let obs_space = vecenv.observation_space();
        let action_space = vecenv.action_space();
        let obs_size = obs_space.shape().iter().product::<usize>() as i64;
        let action_size = match action_space {
            crate::spaces::DynSpace::Discrete(_) => 1,
            crate::spaces::DynSpace::Box(b) => b.shape().iter().product::<usize>() as i64,
            _ => 1,
        };
        let num_envs = vecenv.num_envs();

        let optimizer = nn::Adam::default()
            .build(policy.var_store_mut(), config.learning_rate)
            .expect("Failed to create optimizer");

        let buffer = ExperienceBuffer::new(
            config.batch_size / num_envs,
            num_envs,
            obs_size,
            action_size,
            device,
        );

        let state = policy.initial_state(num_envs as i64);

        // Lagrangian multiplier
        let vs_lagrange = nn::VarStore::new(Device::Cpu);
        let lagrangian = vs_lagrange.root().var(
            "lambda",
            &[],
            nn::Init::Const(safe_config.initial_lagrangian),
        );
        let lagrangian_optimizer = nn::Adam::default()
            .build(&vs_lagrange, safe_config.lagrangian_lr)
            .unwrap();

        let progress = if config.total_timesteps > 0 {
            let pb = ProgressBar::new(config.total_timesteps);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                    .unwrap(),
            );
            Some(pb)
        } else {
            None
        };

        Self {
            config,
            safe_config,
            vecenv,
            policy,
            optimizer,
            buffer,
            global_step: 0,
            epoch: 0,
            start_time: Instant::now(),
            num_envs,
            obs_size,
            action_size,
            state,
            progress,
            lagrangian,
            lagrangian_optimizer,
            last_loss: 0.0,
            last_cost: 0.0,
            mean_reward: 0.0,
        }
    }

    pub fn train(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let (obs, _) = self.vecenv.reset(Some(self.config.seed));

        while self.global_step < self.config.total_timesteps {
            self.collect_rollout(&obs);

            let obs_tensor = match obs {
                ObservationBatch::Cpu(ref a) => Tensor::from_slice(a.as_slice().unwrap())
                    .reshape([self.num_envs as i64, self.obs_size])
                    .to_device(self.config.device),
                #[cfg(feature = "torch")]
                ObservationBatch::Torch(ref t) => t.to_device(self.config.device),
            };
            let (_, last_value, last_cost_value, _) =
                self.policy.forward_safe(&obs_tensor, &self.state);

            self.buffer.compute_returns_and_advantages(
                &last_value,
                self.config.gamma,
                self.config.gae_lambda,
            );

            self.buffer.compute_cost_advantages(
                &last_cost_value,
                self.config.gamma,
                self.config.gae_lambda,
            );

            self.update();

            self.epoch += 1;
            if let Some(ref pb) = self.progress {
                pb.set_position(self.global_step);
                let sps = self.global_step as f64 / self.start_time.elapsed().as_secs_f64();
                pb.set_message(format!(
                    "Loss: {:.4} Cost: {:.2} Reward: {:.2} Lambda: {:.4} SPS: {:.2}",
                    self.last_loss,
                    self.last_cost,
                    self.mean_reward,
                    self.lagrangian.double_value(&[]),
                    sps
                ));
            }
        }
        Ok(())
    }

    fn collect_rollout(&mut self, initial_obs: &ObservationBatch) {
        self.buffer.reset();
        let mut obs = initial_obs.clone();
        let steps_per_env = self.config.batch_size / self.num_envs;

        for _ in 0..steps_per_env {
            let obs_tensor = match obs {
                ObservationBatch::Cpu(ref a) => Tensor::from_slice(a.as_slice().unwrap())
                    .reshape([self.num_envs as i64, self.obs_size])
                    .to_device(self.config.device),
                #[cfg(feature = "torch")]
                ObservationBatch::Torch(ref t) => t.to_device(self.config.device),
            };

            let (dist, value, cost_value, next_state) =
                self.policy.forward_safe(&obs_tensor, &self.state);
            let action_sample = dist.sample();
            let log_prob_sample = dist.log_prob(&action_sample);
            let DistributionSample::Torch(action) = action_sample;
            let DistributionSample::Torch(log_prob) = log_prob_sample;

            let action_vec: Vec<f32> =
                Vec::try_from(action.to_device(Device::Cpu).flatten(0, -1)).unwrap();
            let action_array = ndarray::Array2::from_shape_vec(
                (self.num_envs, self.action_size as usize),
                action_vec,
            )
            .unwrap();

            let result = self.vecenv.step(&action_array);
            let rewards = Tensor::from_slice(&result.rewards);
            let costs = Tensor::from_slice(&result.costs);

            self.mean_reward = result.rewards.iter().sum::<f32>() as f64 / self.num_envs as f64;
            self.last_cost = result.costs.iter().sum::<f32>() as f64 / self.num_envs as f64;

            let dones: Vec<f32> = result
                .dones()
                .iter()
                .map(|&d| if d { 1.0 } else { 0.0 })
                .collect();
            let dones_tensor = Tensor::from_slice(&dones);

            self.buffer.add(
                &obs_tensor.detach(),
                &action
                    .detach()
                    .reshape([self.num_envs as i64, self.action_size]),
                &log_prob.detach(),
                &rewards,
                &dones_tensor,
                &value.detach(),
                &costs,
                &cost_value.detach(),
            );

            obs = result.observations;
            self.state = next_state;
            self.global_step += self.num_envs as u64;
        }
    }

    fn update(&mut self) {
        let minibatch_size = self.config.minibatch_size();
        let lambda = self.lagrangian.double_value(&[]);

        for _ in 0..self.config.update_epochs {
            for _ in 0..self.config.num_minibatches {
                let indices = self.buffer.get_minibatch_indices(minibatch_size);
                let batch = self.buffer.get_minibatch(&indices);

                let (dist, values, cost_values, _) =
                    self.policy.forward_safe(&batch.observations, &None);

                let action_sample = DistributionSample::Torch(batch.actions.shallow_clone());
                let new_log_probs_sample = dist.log_prob(&action_sample);
                let DistributionSample::Torch(new_log_probs) = new_log_probs_sample;

                // Normalize advantages
                let adv = &batch.advantages;
                let normalized_adv = (adv - adv.mean(Kind::Float)) / (adv.std(false) + 1e-8);

                let cost_adv = &batch.cost_advantages;
                let normalized_cost_adv =
                    (cost_adv - cost_adv.mean(Kind::Float)) / (cost_adv.std(false) + 1e-8);

                // Safe Objective: J = Rewards_Adv - Lambda * Cost_Adv
                let combined_adv = &normalized_adv - lambda * &normalized_cost_adv;

                let policy_loss = ppo_policy_loss(
                    &combined_adv,
                    &new_log_probs,
                    &batch.log_probs,
                    self.config.clip_coef,
                );
                let value_loss = ppo_value_loss(
                    &values,
                    &batch.values,
                    &batch.returns,
                    self.config.vf_clip_coef,
                );
                let cost_value_loss = ppo_value_loss(
                    &cost_values,
                    &batch.cost_values,
                    &batch.cost_returns,
                    self.config.vf_clip_coef,
                );

                let loss = &policy_loss + self.config.vf_coef * (&value_loss + &cost_value_loss);

                self.optimizer.zero_grad();
                loss.backward();
                self.optimizer.step();

                self.last_loss = loss.double_value(&[]);
            }
        }

        // Update Lagrangian: lambda = max(0, lambda + lr * (batch_cost - limit))
        let batch_cost = self.buffer.costs.mean(Kind::Float).double_value(&[]);
        let lagrange_loss =
            -self.lagrangian.shallow_clone() * (batch_cost - self.safe_config.cost_limit as f64);

        self.lagrangian_optimizer.zero_grad();
        lagrange_loss.backward();
        self.lagrangian_optimizer.step();

        // Project back to non-negative
        tch::no_grad(|| {
            self.lagrangian.copy_(&self.lagrangian.clamp_min(0.0));
        });
    }
}
