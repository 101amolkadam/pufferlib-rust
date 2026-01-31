//! Exploration modules for RL agents.

#[cfg(feature = "torch")]
use tch::{nn, nn::Module, Kind, Tensor};

/// Intrinsic Curiosity Module (ICM)
///
/// Based on "Curiosity-driven Exploration by Self-supervised Prediction"
/// (Pathak et al., 2017)
#[cfg(feature = "torch")]
pub struct ICM {
    encoder: nn::Sequential,
    forward_model: nn::Sequential,
    inverse_model: nn::Sequential,
    feature_size: i64,
}

#[cfg(feature = "torch")]
impl ICM {
    pub fn new(
        vs: &nn::Path,
        obs_size: i64,
        action_size: i64,
        feature_size: i64,
        hidden_size: i64,
    ) -> Self {
        let encoder = nn::seq()
            .add(nn::linear(
                vs / "encoder_0",
                obs_size,
                hidden_size,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                vs / "encoder_1",
                hidden_size,
                feature_size,
                Default::default(),
            ));

        let forward_model = nn::seq()
            .add(nn::linear(
                vs / "forward_0",
                feature_size + action_size,
                hidden_size,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                vs / "forward_1",
                hidden_size,
                feature_size,
                Default::default(),
            ));

        let inverse_model = nn::seq()
            .add(nn::linear(
                vs / "inverse_0",
                feature_size * 2,
                hidden_size,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                vs / "inverse_1",
                hidden_size,
                action_size,
                Default::default(),
            ));

        Self {
            encoder,
            forward_model,
            inverse_model,
            feature_size,
        }
    }

    /// Compute intrinsic reward and losses
    pub fn compute_intrinsic_reward(
        &self,
        obs: &Tensor,
        next_obs: &Tensor,
        action: &Tensor,
    ) -> (Tensor, Tensor, Tensor) {
        let phi_s = self.encoder.forward(obs);
        let phi_s_next = self.encoder.forward(next_obs);

        // Forward model: predict phi(s_t+1) from phi(s_t) and a_t
        let forward_input = Tensor::cat(&[phi_s.shallow_clone(), action.shallow_clone()], -1);
        let phi_s_next_pred = self.forward_model.forward(&forward_input);

        let forward_loss = phi_s_next_pred
            .mse_loss(&phi_s_next, tch::Reduction::None)
            .mean_dim(Some(&[-1][..]), false, Kind::Float);
        let intrinsic_reward = forward_loss.shallow_clone();

        // Inverse model: predict a_t from phi(s_t) and phi(s_t+1)
        let inverse_input = Tensor::cat(&[phi_s, phi_s_next], -1);
        let action_pred = self.inverse_model.forward(&inverse_input);

        // Assuming discrete actions for cross_entropy, or MSE for continuous
        // For now, let's use MSE as a generic approach if action is already scaled/one-hot
        let inverse_loss = action_pred.mse_loss(action, tch::Reduction::Mean);

        (
            intrinsic_reward,
            forward_loss.mean(Kind::Float),
            inverse_loss,
        )
    }
}

/// Random Network Distillation (RND)
///
/// Based on "Exploration by Random Network Distillation"
/// (Burda et al., 2018)
#[cfg(feature = "torch")]
pub struct RND {
    predictor: nn::Sequential,
    target: nn::Sequential,
    obs_size: i64,
    out_size: i64,
}

#[cfg(feature = "torch")]
impl RND {
    pub fn new(vs: &nn::Path, obs_size: i64, out_size: i64, hidden_size: i64) -> Self {
        // Predictor is trained to match the output of the fixed target network
        let predictor = nn::seq()
            .add(nn::linear(
                vs / "predictor_0",
                obs_size,
                hidden_size,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                vs / "predictor_1",
                hidden_size,
                hidden_size,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                vs / "predictor_2",
                hidden_size,
                out_size,
                Default::default(),
            ));

        // Target network is fixed and random
        let target = nn::seq()
            .add(nn::linear(
                vs / "target_0",
                obs_size,
                hidden_size,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                vs / "target_1",
                hidden_size,
                out_size,
                Default::default(),
            ));

        Self {
            predictor,
            target,
            obs_size,
            out_size,
        }
    }

    /// Compute RND intrinsic reward (prediction error)
    pub fn compute_intrinsic_reward(&self, obs: &Tensor) -> (Tensor, Tensor) {
        let target_out = self.target.forward(obs);
        let predictor_out = self.predictor.forward(obs);

        // Intrinsic reward is the MSE between predictor and target
        let error = predictor_out
            .mse_loss(&target_out, tch::Reduction::None)
            .mean_dim(Some(&[-1][..]), false, Kind::Float);

        let loss = error.mean(Kind::Float);

        (error, loss)
    }
}

/// Hindsight Experience Replay (HER)
///
/// Based on "Hindsight Experience Replay"
/// (Andrychowicz et al., 2017)
pub struct HER {
    /// Strategy for selecting hindsight goals (currently only 'last')
    pub goal_selection_strategy: String,
    /// Probability of relabeling a transition
    pub relabel_prob: f32,
}

impl HER {
    pub fn new(strategy: &str, relabel_prob: f32) -> Self {
        Self {
            goal_selection_strategy: strategy.to_string(),
            relabel_prob,
        }
    }

    /// Note: HER usually operates on the replay buffer.
    /// In PPO (on-policy), HER is typically applied by relabeling
    /// within the current rollout or using a specialized Goal-Conditioned buffer.
    pub fn relabel_trajectory(
        &self,
        observations: &mut [ndarray::ArrayD<f32>],
        _actions: &[ndarray::ArrayD<f32>],
        rewards: &mut [f32],
        _goals: &[ndarray::ArrayD<f32>],
    ) {
        if observations.is_empty() {
            return;
        }

        // Simple 'last' strategy: the last state in trajectory is used as goal
        let hindsight_goal = observations.last().unwrap().clone();

        for i in 0..observations.len() {
            if rand::random::<f32>() < self.relabel_prob {
                // Relabel reward: if current observation matches hindsight goal
                // This is a simplified check, usually requires a distance function
                let dist = (&observations[i] - &hindsight_goal).mapv(|x| x * x).sum();
                if dist < 1e-3 {
                    rewards[i] = 1.0;
                } else {
                    rewards[i] = 0.0;
                }
                // In a real implementation, we would also update the 'goal' part
                // of the observation if it's concatenated.
            }
        }
    }
}
