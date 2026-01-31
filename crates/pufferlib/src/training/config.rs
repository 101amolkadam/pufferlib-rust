//! Trainer configuration.

use serde::{Deserialize, Serialize};
#[cfg(feature = "torch")]
use tch::Device;

/// Configuration for the PPO trainer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainerConfig {
    // Training
    /// Total timesteps to train
    pub total_timesteps: u64,
    /// Number of steps per rollout
    pub batch_size: usize,
    /// Horizon for BPTT
    pub bptt_horizon: usize,
    /// Number of minibatches per update
    pub num_minibatches: usize,
    /// Number of epochs per batch
    pub update_epochs: usize,

    // PPO hyperparameters
    /// Learning rate
    pub learning_rate: f64,
    /// Discount factor
    pub gamma: f64,
    /// GAE lambda
    pub gae_lambda: f64,
    /// PPO clipping coefficient
    pub clip_coef: f64,
    /// Value function clipping coefficient
    pub vf_clip_coef: f64,
    /// Entropy coefficient
    pub ent_coef: f64,
    /// Value function coefficient
    pub vf_coef: f64,
    /// Maximum gradient norm
    pub max_grad_norm: f64,
    /// Target KL divergence for early stopping
    pub target_kl: f64,
    /// KL coefficient for adaptive penalty
    pub kl_coef: f64,
    /// Whether to use adaptive KL penalty
    pub kl_adaptive: bool,
    /// PPO dual-clip coefficient (0.0 to disable)
    pub dual_clip_coef: f64,

    // V-trace
    /// V-trace rho clipping
    pub vtrace_rho_clip: f64,
    /// V-trace c clipping
    /// V-trace c clipping
    pub vtrace_c_clip: f64,
    /// Whether to use V-trace for advantage estimation
    pub use_vtrace: bool,

    // Optimization
    /// Whether to anneal learning rate
    pub anneal_lr: bool,
    /// Minimum LR ratio for annealing
    pub min_lr_ratio: f64,

    // Checkpointing
    /// Checkpoint interval (epochs)
    pub checkpoint_interval: usize,
    /// Data directory for checkpoints
    pub data_dir: String,

    // Device
    /// Device to train on ("cpu" or "cuda")
    #[cfg(feature = "torch")]
    #[serde(skip, default = "default_device")]
    pub device: Device,

    // Self-play
    /// Enable self-play
    pub self_play_enabled: bool,
    /// Learner ratio (fraction of envs that run the learner policy)
    pub self_play_learner_ratio: f64,
    /// Snapshot interval (epochs)
    pub self_play_snapshot_interval: usize,

    // Logging
    /// Whether to enable W&B logging
    pub wandb_enabled: bool,
    /// W&B project name
    pub wandb_project: String,
    /// W&B run name
    pub wandb_run_name: String,
    /// W&B entity name
    pub wandb_entity: String,
    /// W&B group name
    pub wandb_group: String,

    // AMP
    /// Whether to use Automatic Mixed Precision (AMP)
    pub use_amp: bool,
    /// Initial scale for GradScaler
    pub amp_initial_scale: f64,

    // Performance
    /// Number of minibatches to accumulate gradients over
    pub gradient_accumulation_steps: usize,

    // ICM
    /// Weight for intrinsic rewards
    pub icm_beta: f64,

    // RND
    /// Weight for RND intrinsic rewards
    pub rnd_beta: f64,

    // Random seed
    pub seed: u64,
}

/// Configuration for Safe RL (Constrained PPO)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConstrainedPpoConfig {
    /// Safety threshold for average cost per episode or per step (depends on impl)
    /// Usually interpreted as the limit on the expected cumulative cost.
    pub cost_limit: f32,
    /// Learning rate for the Lagrangian multiplier
    pub lagrangian_lr: f64,
    /// Initial value for the Lagrangian multiplier (λ)
    pub initial_lagrangian: f64,
    /// Whether to use GAE for cost advantage estimation
    pub use_cost_gae: bool,
}

impl Default for ConstrainedPpoConfig {
    fn default() -> Self {
        Self {
            cost_limit: 1.0,
            lagrangian_lr: 0.01,
            initial_lagrangian: 0.1,
            use_cost_gae: true,
        }
    }
}

#[cfg(feature = "torch")]
fn default_device() -> Device {
    Device::Cpu
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            total_timesteps: 10_000_000,
            batch_size: 8192,
            bptt_horizon: 64,
            num_minibatches: 4,
            update_epochs: 1,

            learning_rate: 0.0003,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_coef: 0.2,
            vf_clip_coef: 0.2,
            ent_coef: 0.01,
            vf_coef: 0.5,
            max_grad_norm: 0.5,
            target_kl: 0.015,
            kl_coef: 0.2,
            kl_adaptive: true,
            dual_clip_coef: 0.0,

            vtrace_rho_clip: 1.0,
            vtrace_c_clip: 1.0,
            use_vtrace: false,

            anneal_lr: true,
            min_lr_ratio: 0.0,

            checkpoint_interval: 100,
            data_dir: std::env::var("PUFFER_DATA_DIR")
                .unwrap_or_else(|_| "checkpoints".to_string()),

            self_play_enabled: false,
            self_play_learner_ratio: 1.0,
            self_play_snapshot_interval: 100,

            wandb_enabled: std::env::var("WANDB_ENABLED")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false),
            wandb_project: std::env::var("WANDB_PROJECT")
                .unwrap_or_else(|_| "pufferlib".to_string()),
            wandb_run_name: std::env::var("WANDB_RUN_NAME").unwrap_or_else(|_| "ppo".to_string()),
            wandb_entity: std::env::var("WANDB_ENTITY").unwrap_or_default(),
            wandb_group: std::env::var("WANDB_GROUP").unwrap_or_default(),

            use_amp: false,
            amp_initial_scale: 65536.0,

            #[cfg(feature = "torch")]
            device: Device::Cpu,
            gradient_accumulation_steps: 1,
            icm_beta: 0.1,
            rnd_beta: 0.1,
            seed: 42,
        }
    }
}

impl TrainerConfig {
    /// Create config for CUDA device
    #[cfg(feature = "torch")]
    pub fn cuda(mut self) -> Self {
        self.device = Device::Cuda(0);
        self
    }

    /// Set total timesteps
    pub fn with_timesteps(mut self, timesteps: u64) -> Self {
        self.total_timesteps = timesteps;
        self
    }

    /// Set learning rate
    pub fn with_lr(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Enable W&B logging
    pub fn with_wandb(mut self, project: &str, name: &str) -> Self {
        self.wandb_enabled = true;
        self.wandb_project = project.to_string();
        self.wandb_run_name = name.to_string();
        self
    }

    /// Compute minibatch size
    pub fn minibatch_size(&self) -> usize {
        self.batch_size / self.num_minibatches
    }
}
