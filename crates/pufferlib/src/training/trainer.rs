//! Main PPO trainer.

use super::buffer::ExperienceBuffer;
use super::config::TrainerConfig;
use super::ppo::{ppo_policy_loss, ppo_value_loss};
use crate::policy::{HasVarStore, Policy};
use crate::vector::VecEnvBackend;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

/// Main trainer for PPO algorithm
pub struct Trainer<P: Policy + HasVarStore, V: VecEnvBackend> {
    /// Configuration
    config: TrainerConfig,
    /// Vector environment
    vecenv: V,
    /// Policy network
    policy: P,
    /// Optimizer
    optimizer: nn::Optimizer,
    /// Experience buffer
    buffer: ExperienceBuffer,
    /// Current global step
    global_step: u64,
    /// Current epoch
    epoch: u64,
    /// Start time
    start_time: Instant,
    /// Number of environments
    num_envs: usize,
    /// Observation size
    obs_size: i64,
    /// Action size
    action_size: i64,
    /// Current policy state (for recurrent policies)
    state: Option<Vec<Tensor>>,
    /// Progress bar
    progress: Option<ProgressBar>,
}

impl<P: Policy + HasVarStore, V: VecEnvBackend> Trainer<P, V> {
    /// Create a new trainer
    pub fn new(vecenv: V, mut policy: P, config: TrainerConfig, _device: Device) -> Self {
        let obs_space = vecenv.observation_space();
        let _action_space = vecenv.action_space();

        let obs_size = obs_space.shape().iter().product::<usize>() as i64;
        // Assuming discrete action space for now as per PPO implementation details in buffer
        let action_size = 1;
        let num_envs = vecenv.num_envs();

        let optimizer = nn::Adam::default()
            .build(policy.var_store_mut(), config.learning_rate)
            .expect("Failed to create optimizer");

        // Note: Buffer initialization logic might need to adjust based on space types
        let buffer = ExperienceBuffer::new(
            config.batch_size / num_envs,
            num_envs,
            obs_size,
            action_size,
            config.device,
        );

        let state = policy.initial_state(num_envs as i64);

        let progress = if config.total_timesteps > 0 {
            let pb = ProgressBar::new(config.total_timesteps);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                    .unwrap()
                    .progress_chars("#>-"),
            );
            Some(pb)
        } else {
            None
        };

        Self {
            config,
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
        }
    }

    /// Run the training loop
    pub fn train(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let (mut obs, _) = self.vecenv.reset(Some(self.config.seed));

        while self.global_step < self.config.total_timesteps {
            // Collect rollout
            self.collect_rollout(&obs);

            // Update obs from last step
            let total_steps = self.buffer.len() as i64;
            let last_obs = self
                .buffer
                .observations
                .narrow(0, total_steps - self.num_envs as i64, self.num_envs as i64)
                .reshape([self.num_envs as i64, self.obs_size]);
            obs = ndarray::Array2::from_shape_vec(
                (self.num_envs, self.obs_size as usize),
                Vec::<f32>::try_from(last_obs.flatten(0, -1)).unwrap(),
            )
            .unwrap();

            // Compute returns and advantages
            let obs_tensor = Tensor::from_slice(obs.as_slice().unwrap())
                .reshape([self.num_envs as i64, self.obs_size])
                .to_device(self.config.device);
            let (_, last_value, _) = self.policy.forward(&obs_tensor, &self.state);

            if self.config.use_vtrace {
                self.buffer.compute_vtrace(
                    &last_value,
                    self.config.gamma,
                    self.config.gae_lambda,
                    self.config.vtrace_rho_clip,
                    self.config.vtrace_c_clip,
                );
            } else {
                self.buffer.compute_returns_and_advantages(
                    &last_value,
                    self.config.gamma,
                    self.config.gae_lambda,
                );
            }

            // PPO update
            self.update();

            self.epoch += 1;

            // Increment global step and update progress bar
            if let Some(ref pb) = self.progress {
                pb.set_position(self.global_step);
                let elapsed = self.start_time.elapsed().as_secs_f64();
                let sps = self.global_step as f64 / elapsed;
                pb.set_message(format!("SPS: {:.2}", sps));
            }

            // Logging
            if self.epoch % 10 == 0 {
                let elapsed = self.start_time.elapsed().as_secs_f64();
                let sps = self.global_step as f64 / elapsed;
                if self.progress.is_none() {
                    tracing::info!(
                        step = self.global_step,
                        epoch = self.epoch,
                        sps = sps,
                        "Training progress"
                    );
                }
            }

            // Checkpointing
            if self.epoch > 0 && self.epoch % self.config.checkpoint_interval as u64 == 0 {
                self.save_checkpoint();
            }
        }

        if let Some(ref pb) = self.progress {
            pb.finish_with_message("Training complete");
        }

        Ok(())
    }

    /// Collect a rollout of experience
    fn collect_rollout(&mut self, initial_obs: &ndarray::Array2<f32>) {
        self.buffer.reset();

        let mut obs_array = initial_obs.clone();
        let steps_per_env = self.config.batch_size / self.num_envs;

        for _ in 0..steps_per_env {
            // Convert obs to tensor
            let obs_tensor = Tensor::from_slice(obs_array.as_slice().unwrap())
                .reshape([self.num_envs as i64, self.obs_size])
                .to_device(self.config.device);

            // Get action from policy
            let (logits, value, next_state) = self.policy.forward(&obs_tensor, &self.state);
            let action = self.policy.sample_actions(&logits);
            let log_prob = self.policy.log_probs(&logits, &action);

            // Convert action to ndarray
            let action_vec: Vec<f32> =
                Vec::try_from(action.to_device(Device::Cpu).flatten(0, -1)).unwrap();
            let action_array = ndarray::Array2::from_shape_vec(
                (self.num_envs, self.action_size as usize),
                action_vec,
            )
            .unwrap();

            // Step environment
            let result = self.vecenv.step(&action_array);

            // Store in buffer
            let rewards = Tensor::from_slice(&result.rewards);
            let dones_bool = result.dones();
            let dones: Vec<f32> = dones_bool
                .iter()
                .map(|&d| if d { 1.0 } else { 0.0 })
                .collect();
            let dones_tensor = Tensor::from_slice(&dones);

            self.buffer.add(
                &obs_tensor,
                &action.unsqueeze(-1).to_kind(Kind::Float),
                &log_prob,
                &rewards,
                &dones_tensor,
                &value,
            );

            // Update obs for next step
            obs_array = result.observations;

            // Update state
            if let Some(states) = next_state {
                let dones_dev = Tensor::from_slice(
                    &dones
                        .iter()
                        .map(|&d| if d > 0.5 { 0.0 } else { 1.0 })
                        .collect::<Vec<f32>>(),
                )
                .to_device(self.config.device)
                .reshape([1, self.num_envs as i64, 1]);

                self.state = Some(states.into_iter().map(|s| s * &dones_dev).collect());
            } else {
                self.state = None;
            }

            self.global_step += self.num_envs as u64;
        }
    }

    /// Perform PPO update on collected experience
    fn update(&mut self) {
        let minibatch_size = self.config.minibatch_size();

        for _ in 0..self.config.update_epochs {
            let indices = self.buffer.get_minibatch_indices(minibatch_size);
            let batch = self.buffer.get_minibatch(&indices);

            // Normalize advantages
            let advantages = &batch.advantages;
            let adv_mean = advantages.mean(Kind::Float);
            let adv_std = advantages.std(false);
            let normalized_advantages = (advantages - &adv_mean) / (&adv_std + 1e-8);

            // Forward pass
            let (logits, values, _) = self.policy.forward(&batch.observations, &None);
            let actions = batch.actions.to_kind(Kind::Int64).squeeze_dim(-1);
            let new_log_probs = self.policy.log_probs(&logits, &actions);
            let entropy = self.policy.entropy(&logits);

            // Compute losses
            let policy_loss = ppo_policy_loss(
                &normalized_advantages,
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

            let entropy_loss = -entropy.mean(Kind::Float);

            let loss = &policy_loss
                + self.config.vf_coef * &value_loss
                + self.config.ent_coef * &entropy_loss;

            // Backward pass
            self.optimizer.zero_grad();
            loss.backward();

            // Gradient clipping
            // Note: tch-rs doesn't have a direct clip_grad_norm, so we skip for now

            self.optimizer.step();
        }
    }

    /// Save checkpoint
    /// Save checkpoint
    fn save_checkpoint(&self) {
        let start_time = Instant::now();
        if let Err(e) = std::fs::create_dir_all(&self.config.data_dir) {
            tracing::error!("Failed to create checkpoint directory: {}", e);
            return;
        }

        let path = format!("{}/checkpoint_{:06}.pt", self.config.data_dir, self.epoch);
        tracing::info!(path = %path, "Saving checkpoint");

        if let Err(e) = self.policy.var_store().save(&path) {
            tracing::error!("Failed to save checkpoint: {}", e);
        } else {
            // also save optimizer? tch-rs doesn't easily support saving optimizer state separately properly without some work,
            // usually one saves everything in the varstore if optimizer variables are registered there.
            // But PyTorch standard is saving state dicts.
            // VarStore saves network weights. Optimizer state is separate.
            // For now saving policy weights is the most important part.
            tracing::info!(elapsed = ?start_time.elapsed(), "Checkpoint saved");
        }
    }

    /// Get current global step
    pub fn global_step(&self) -> u64 {
        self.global_step
    }

    /// Get current epoch
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Get samples per second
    pub fn sps(&self) -> f64 {
        self.global_step as f64 / self.start_time.elapsed().as_secs_f64()
    }
}
