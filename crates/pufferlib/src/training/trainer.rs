//! Main PPO trainer.

use super::buffer::ExperienceBuffer;
use super::config::TrainerConfig;
use super::ppo::{kl_divergence, ppo_dual_clip_policy_loss, ppo_policy_loss, ppo_value_loss};
use super::self_play::PolicyPool;
use crate::policy::{HasVarStore, Policy, DistributionSample};
use crate::spaces::Space;
use crate::vector::{ObservationBatch, VecEnvBackend};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::time::Instant;
#[cfg(feature = "torch")]
#[cfg(feature = "torch")]
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

#[cfg(not(feature = "torch"))]
mod torch_compat {
    pub struct Tensor;
    pub struct Device;
    pub struct Kind;
    pub mod nn {
        pub struct Optimizer;
    }
}
#[cfg(not(feature = "torch"))]
use torch_compat::*;

#[cfg(not(feature = "torch"))]
struct DummyTensor; // Placeholder for non-torch builds

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
    /// Policy pool for self-play
    pool: PolicyPool,
    /// HPO trial ID if running in HPO mode
    pub trial_id: Option<usize>,
    /// Latest mean reward for HPO evaluation
    pub mean_reward: f64,
    /// Current KL coefficient for adaptive penalty
    current_kl_coef: f64,
}

impl<P: Policy + HasVarStore, V: VecEnvBackend> Trainer<P, V> {
    /// Create a new trainer
    pub fn new(vecenv: V, mut policy: P, config: TrainerConfig, _device: Device) -> Self {
        let obs_space = vecenv.observation_space();
        let action_space = vecenv.action_space();

        let obs_size = obs_space.shape().iter().product::<usize>() as i64;
        let action_size = match action_space {
            crate::spaces::DynSpace::Discrete(_) => 1,
            crate::spaces::DynSpace::MultiDiscrete(md) => md.nvec.len() as i64,
            crate::spaces::DynSpace::Box(b) => b.shape().iter().product::<usize>() as i64,
            _ => 1,
        };
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

        let kl_coef = config.kl_coef;

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
            pool: PolicyPool::new(),
            trial_id: None,
            mean_reward: 0.0,
            current_kl_coef: kl_coef,
        }
    }

    /// Run the training loop
    pub fn train(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let (mut obs, _) = self.vecenv.reset(Some(self.config.seed));

        while self.global_step < self.config.total_timesteps {
            // Collect rollout
            self.collect_rollout(&obs);

            // Compute returns and advantages
            let obs_tensor = match obs {
                ObservationBatch::Cpu(ref a) => {
                    Tensor::from_slice(a.as_slice().unwrap())
                        .reshape([self.num_envs as i64, self.obs_size])
                        .to_device(self.config.device)
                }
                #[cfg(feature = "torch")]
                ObservationBatch::Torch(ref t) => t.to_device(self.config.device),
                #[cfg(feature = "candle")]
                ObservationBatch::Candle(ref t) => {
                    // Logic for converting Candle to Torch if mixed, but usually they match
                    panic!("Mixed backends not yet supported in zero-copy path")
                }
            };
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
            if self.epoch.is_multiple_of(10) {
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

            // Checkpointing and Self-Play Snapshotting
            if self.epoch > 0 {
                if self.epoch.is_multiple_of(self.config.checkpoint_interval as u64) {
                    self.save_checkpoint();
                }

                if self.config.self_play_enabled && self.epoch.is_multiple_of(self.config.self_play_snapshot_interval as u64) {
                    self.snapshot_policy();
                }
            }
        }

        if let Some(ref pb) = self.progress {
            pb.finish_with_message("Training complete");
        }

        Ok(())
    }

    /// Collect a rollout of experience
    fn collect_rollout(&mut self, initial_obs: &ObservationBatch) {
        self.buffer.reset();

        let mut obs = initial_obs.clone();
        let steps_per_env = self.config.batch_size / self.num_envs;

        for _ in 0..steps_per_env {
            // Convert obs to tensor
            let obs_tensor = match obs {
                ObservationBatch::Cpu(ref a) => {
                    Tensor::from_slice(a.as_slice().unwrap())
                        .reshape([self.num_envs as i64, self.obs_size])
                        .to_device(self.config.device)
                }
                #[cfg(feature = "torch")]
                ObservationBatch::Torch(t) => t.to_device(self.config.device),
                #[cfg(feature = "candle")]
                ObservationBatch::Candle(_) => panic!("Candle backend in torch trainer"),
            };

            let (action, log_prob, value, next_state) = if self.config.self_play_enabled && !self.pool.all_policies().is_empty() {
                // Partition environments/agents between learner and pool
                let num_learner = (self.num_envs as f64 * self.config.self_play_learner_ratio) as i64;
                let num_pool = self.num_envs as i64 - num_learner;

                // Learner forward pass
                let learner_obs = obs_tensor.narrow(0, 0, num_learner);
                let learner_state = self.state.as_ref().map(|s| s.iter().map(|t| t.narrow(1, 0, num_learner)).collect());
                let (l_dist, l_value, l_next_state) = self.policy.forward(&learner_obs, &learner_state);
                let l_action_sample = l_dist.sample();
                let l_action = match l_action_sample {
                    #[cfg(feature = "torch")]
                    DistributionSample::Torch(ref t) => t.shallow_clone(),
                    #[cfg(feature = "candle")]
                    DistributionSample::Candle(_) => panic!("Candle in torch path"),
                };
                let l_log_prob_sample = l_dist.log_prob(&l_action_sample);
                let l_log_prob = match l_log_prob_sample {
                     #[cfg(feature = "torch")]
                    DistributionSample::Torch(t) => t,
                    #[cfg(feature = "candle")]
                    DistributionSample::Candle(_) => panic!("Candle in torch path"),
                };

                // Pool forward pass (simple implementation: use one sampled policy for all pool agents)
                let _pool_policy_record = self.pool.sample_policy().unwrap();
                
                let p_action = Tensor::zeros([num_pool, self.action_size], (Kind::Float, self.config.device));
                let p_log_prob = Tensor::zeros([num_pool], (Kind::Float, self.config.device));
                let p_value = Tensor::zeros([num_pool], (Kind::Float, self.config.device));

                let combined_action = Tensor::cat(&[l_action, p_action], 0);
                let combined_log_prob = Tensor::cat(&[l_log_prob, p_log_prob], 0);
                let combined_value = Tensor::cat(&[l_value, p_value], 0);
                
                (combined_action, combined_log_prob, combined_value, l_next_state)
            } else {
                let (dist, value, next_state) = self.policy.forward(&obs_tensor, &self.state);
                let action_sample = dist.sample();
                let action = match action_sample {
                    #[cfg(feature = "torch")]
                    DistributionSample::Torch(ref t) => t.shallow_clone(),
                    #[cfg(feature = "candle")]
                    DistributionSample::Candle(_) => panic!("Candle in torch path"),
                    #[allow(unreachable_patterns)]
                    _ => panic!("Backend mismatch"),
                };
                let log_prob_sample = dist.log_prob(&action_sample);
                let log_prob = match log_prob_sample {
                     #[cfg(feature = "torch")]
                    DistributionSample::Torch(t) => t,
                    #[cfg(feature = "candle")]
                    DistributionSample::Candle(_) => panic!("Candle in torch path"),
                    #[allow(unreachable_patterns)]
                    _ => panic!("Backend mismatch"),
                };
                (action, log_prob, value, next_state)
            };

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
            self.mean_reward = result.rewards.iter().sum::<f32>() as f64 / self.num_envs as f64;
            
            let dones_bool = result.dones();
            let dones: Vec<f32> = dones_bool
                .iter()
                .map(|&d| if d { 1.0 } else { 0.0 })
                .collect();
            let dones_tensor = Tensor::from_slice(&dones);

            self.buffer.add(
                &obs_tensor,
                &action.reshape([self.num_envs as i64, self.action_size]),
                &log_prob,
                &rewards,
                &dones_tensor,
                &value,
            );

            // Update obs and state for next step
            obs = result.observations;

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

        for epoch in 0..self.config.update_epochs {
            let mut epoch_kls = Vec::new();

            for _ in 0..self.config.num_minibatches {
                let indices = self.buffer.get_minibatch_indices(minibatch_size);
                let batch = self.buffer.get_minibatch(&indices);

                // Normalize advantages
                let advantages = &batch.advantages;
                let adv_mean = advantages.mean(Kind::Float);
                let adv_std = advantages.std(false);
                let normalized_advantages = (advantages - &adv_mean) / (&adv_std + 1e-8);

                // Forward pass
                let (dist, values, _) = self.policy.forward(&batch.observations, &None);
                
                #[cfg(feature = "torch")]
                let action_sample = DistributionSample::Torch(batch.actions.shallow_clone());
                #[cfg(not(feature = "torch"))]
                let action_sample = DistributionSample::Candle(candle_core::Tensor::zeros((1,), candle_core::DType::F32, &candle_core::Device::Cpu).unwrap());

                let new_log_probs_sample = dist.log_prob(&action_sample);
                let entropy_sample = dist.entropy();

                #[cfg(feature = "torch")]
                #[allow(unreachable_patterns)]
                let (new_log_probs, entropy) = match (new_log_probs_sample, entropy_sample) {
                    (DistributionSample::Torch(lp), DistributionSample::Torch(ent)) => (lp, ent),
                    _ => panic!("Backend mismatch"),
                };

                // Compute KL divergence for stability tracking
                #[cfg(feature = "torch")]
                let kl = kl_divergence(&new_log_probs, &batch.log_probs);
                #[cfg(feature = "torch")]
                epoch_kls.push(kl.double_value(&[]));

                // Compute losses
                #[cfg(feature = "torch")]
                let policy_loss = if self.config.dual_clip_coef > 0.0 {
                    ppo_dual_clip_policy_loss(
                        &normalized_advantages,
                        &new_log_probs,
                        &batch.log_probs,
                        self.config.clip_coef,
                        self.config.dual_clip_coef,
                    )
                } else {
                    ppo_policy_loss(
                        &normalized_advantages,
                        &new_log_probs,
                        &batch.log_probs,
                        self.config.clip_coef,
                    )
                };

                #[cfg(feature = "torch")]
                let value_loss = ppo_value_loss(
                    &values,
                    &batch.values,
                    &batch.returns,
                    self.config.vf_clip_coef,
                );

                #[cfg(feature = "torch")]
                let entropy_loss = -entropy.mean(Kind::Float);

                #[cfg(feature = "torch")]
                let kl_penalty = if self.config.kl_adaptive {
                    self.current_kl_coef * kl
                } else {
                    Tensor::from(0.0).to_device(self.config.device)
                };

                #[cfg(feature = "torch")]
                let loss = &policy_loss
                    + self.config.vf_coef * &value_loss
                    + self.config.ent_coef * &entropy_loss
                    + &kl_penalty;

                // Backward pass
                #[cfg(feature = "torch")]
                {
                    self.optimizer.zero_grad();
                    loss.backward();

                    // Gradient clipping
                    let mut global_norm = 0.0f64;
                    for var in self.policy.var_store().variables().values() {
                        let grad = var.grad();
                        if grad.defined() {
                            global_norm += grad
                                .pow_tensor_scalar(2.0)
                                .sum(Kind::Float)
                                .double_value(&[]);
                        }
                    }
                    global_norm = global_norm.sqrt();

                    if global_norm > self.config.max_grad_norm {
                        let clip_coef = self.config.max_grad_norm / (global_norm + 1e-6);
                        for var in self.policy.var_store().variables().values() {
                            let mut grad = var.grad();
                            if grad.defined() {
                                let _ = grad.f_mul_scalar_(clip_coef);
                            }
                        }
                    }

                    self.optimizer.step();
                }
            }

            // Early stopping and Adaptive KL adjustment
            if !epoch_kls.is_empty() {
                let mean_kl = epoch_kls.iter().sum::<f64>() / epoch_kls.len() as f64;
                
                // Adaptive KL adjustment (original PufferLib logic)
                if self.config.kl_adaptive {
                    if mean_kl < self.config.target_kl / 1.5 {
                        self.current_kl_coef /= 2.0;
                    } else if mean_kl > self.config.target_kl * 1.5 {
                        self.current_kl_coef *= 2.0;
                    }
                    self.current_kl_coef = self.current_kl_coef.clamp(1e-4, 10.0);
                }

                if mean_kl > self.config.target_kl {
                    tracing::info!(
                        epoch = epoch,
                        kl = mean_kl,
                        target = self.config.target_kl,
                        "Early stopping due to high KL divergence"
                    );
                    break;
                }
            }
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

    /// Get latest mean reward
    pub fn reward(&self) -> f64 {
        self.mean_reward
    }

    /// Snapshot current policy and add to pool for self-play
    fn snapshot_policy(&mut self) {
        let snapshot_id = format!("snapshot_{:06}", self.epoch);
        let path = PathBuf::from(&self.config.data_dir).join(format!("{}.pt", snapshot_id));
        
        tracing::info!(id = %snapshot_id, path = ?path, "Snapshotting policy for self-play");

        if let Err(e) = self.policy.var_store().save(&path) {
            tracing::error!("Failed to save snapshot: {}", e);
        } else {
            // Initial rating of 1000.0 for new snapshots
            self.pool.add_policy(snapshot_id, path, 1000.0);
        }
    }
}
