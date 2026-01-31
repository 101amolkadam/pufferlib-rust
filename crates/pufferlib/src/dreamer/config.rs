#[derive(Clone, Debug)]
pub struct DreamerConfig {
    /// Dimension of stochastic state (e.g. 32 classes for categorical).
    pub stoch_size: i64,
    /// Number of categorical classes (if using discrete latent).
    pub stoch_discrete: i64,

    /// Dimension of deterministic state (GRU hidden size).
    pub deter_size: i64,

    /// Encoder/Decoder embedding size.
    pub embedding_size: i64,

    /// Learning rate for model (world model).
    pub model_lr: f64,
    /// Learning rate for actor-critic.
    pub actor_lr: f64,
    pub value_lr: f64,

    /// Discount factor (gamma).
    pub gamma: f64,

    /// GAE lambda (for returns).
    pub lambda: f64,

    /// KL loss scale factor.
    pub kl_scale: f64,
    /// Free nats (minimum KL).
    pub free_nats: f64,

    /// Horizon for imagination (training actor).
    pub horizon: i64,

    /// Horizon for MPC planning.
    pub mpc_horizon: i64,
    /// Number of action sequences to sample per MPC step.
    pub mpc_samples: i64,
    /// Number of refinement iterations (for CEM).
    pub mpc_iterations: i64,
}

impl Default for DreamerConfig {
    fn default() -> Self {
        Self {
            stoch_size: 32,
            stoch_discrete: 32, // 32x32 categorical
            deter_size: 512,
            embedding_size: 1024,
            model_lr: std::env::var("DREAMER_MODEL_LR")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1e-4),
            actor_lr: std::env::var("DREAMER_ACTOR_LR")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8e-5),
            value_lr: std::env::var("DREAMER_VALUE_LR")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8e-5),
            gamma: std::env::var("DREAMER_GAMMA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.99),
            lambda: std::env::var("DREAMER_LAMBDA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.95),
            kl_scale: 1.0,
            free_nats: 1.0,
            horizon: 15,
            mpc_horizon: 10,
            mpc_samples: 512,
            mpc_iterations: 5,
        }
    }
}
