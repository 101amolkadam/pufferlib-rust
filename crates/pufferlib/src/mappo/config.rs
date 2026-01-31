//! MAPPO Configuration.

#[derive(Clone, Debug)]
pub struct MappoConfig {
    /// Number of agents
    pub num_agents: usize,
    /// Whether agents share the same policy network
    pub share_policy: bool,
    /// Whether to use global state for critic (vs concatenated obs)
    pub use_global_state: bool,
    /// Local observation dimension per agent
    pub obs_dim: i64,
    /// Global state dimension (if use_global_state)
    pub global_state_dim: i64,
    /// Action dimension per agent
    pub action_dim: i64,
    /// PPO clip coefficient
    pub clip_coef: f64,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: f64,
    /// Value loss coefficient
    pub vf_coef: f64,
    /// Entropy coefficient
    pub ent_coef: f64,
    /// GAE gamma
    pub gamma: f64,
    /// GAE lambda
    pub gae_lambda: f64,
    /// Number of PPO epochs
    pub update_epochs: i64,
    /// Minibatch size
    pub minibatch_size: Option<i64>,
    /// Learning rate
    pub learning_rate: f64,
}

impl Default for MappoConfig {
    fn default() -> Self {
        Self {
            num_agents: std::env::var("MAPPO_NUM_AGENTS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2),
            share_policy: std::env::var("MAPPO_SHARE_POLICY")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(true),
            use_global_state: std::env::var("MAPPO_USE_GLOBAL_STATE")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(true),
            obs_dim: 16,
            global_state_dim: 32,
            action_dim: 5,
            clip_coef: 0.2,
            max_grad_norm: 0.5,
            vf_coef: 0.5,
            ent_coef: 0.01,
            gamma: std::env::var("MAPPO_GAMMA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.99),
            gae_lambda: std::env::var("MAPPO_GAE_LAMBDA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.95),
            update_epochs: 10,
            minibatch_size: Some(256),
            learning_rate: std::env::var("MAPPO_LEARNING_RATE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3e-4),
        }
    }
}
