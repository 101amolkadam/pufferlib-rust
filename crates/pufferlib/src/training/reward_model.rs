//! Reward modeling for RLHF and RLAIF.

#[cfg(feature = "torch")]
use tch::{nn, nn::Module, Kind, Tensor};

/// Reward model for learning from preferences
#[cfg(feature = "torch")]
pub struct RewardModel {
    model: nn::Sequential,
}

#[cfg(feature = "torch")]
impl RewardModel {
    pub fn new(vs: &nn::Path, obs_size: i64, action_size: i64, hidden_size: i64) -> Self {
        let model = nn::seq()
            .add(nn::linear(
                vs / "layer_0",
                obs_size + action_size,
                hidden_size,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                vs / "layer_1",
                hidden_size,
                hidden_size,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                vs / "layer_2",
                hidden_size,
                1,
                Default::default(),
            ));

        Self { model }
    }

    /// Predict reward for a state-action pair
    pub fn predict_reward(&self, obs: &Tensor, action: &Tensor) -> Tensor {
        let input = Tensor::cat(&[obs, action], -1);
        self.model.forward(&input)
    }

    /// Compute Bradley-Terry preference loss
    ///
    /// preferences: Tensor of shape [Batch], 1.0 if traj_1 > traj_2, 0.0 otherwise
    pub fn compute_preference_loss(
        &self,
        obs_1: &Tensor,
        act_1: &Tensor,
        obs_2: &Tensor,
        act_2: &Tensor,
        preferences: &Tensor,
    ) -> Tensor {
        let r1 =
            self.predict_reward(obs_1, act_1)
                .sum_dim_intlist(Some(&[-1][..]), false, Kind::Float);
        let r2 =
            self.predict_reward(obs_2, act_2)
                .sum_dim_intlist(Some(&[-1][..]), false, Kind::Float);

        // P(traj_1 > traj_2) = exp(sum(r1)) / (exp(sum(r1)) + exp(sum(r2)))
        let logits = r1 - r2;
        logits.binary_cross_entropy_with_logits::<Tensor>(
            preferences,
            None,
            None,
            tch::Reduction::Mean,
        )
    }
}
