use crate::dreamer::config::DreamerConfig;
use crate::dreamer::models::DenseHead;
use crate::dreamer::rssm::{State, RSSM};
use tch::{no_grad, Kind, Tensor};

/// Planner that uses the learned world model to search for optimal actions.
pub struct MPCPlanner<'a> {
    config: &'a DreamerConfig,
    rssm: &'a RSSM,
    reward_head: &'a DenseHead,
}

impl<'a> MPCPlanner<'a> {
    pub fn new(config: &'a DreamerConfig, rssm: &'a RSSM, reward_head: &'a DenseHead) -> Self {
        Self {
            config,
            rssm,
            reward_head,
        }
    }

    /// Plan the next action using the Cross-Entropy Method (CEM).
    /// Returns the first action of the best found sequence.
    pub fn plan(&self, state: &State, action_dim: i64) -> Tensor {
        let device = state.deter.device();
        let samples = self.config.mpc_samples;
        let horizon = self.config.mpc_horizon;

        // Initialize action distribution (mean and std)
        // Shape: [Horizon, ActionDim]
        let mut mean = Tensor::zeros([horizon, action_dim], (Kind::Float, device));
        let mut std = Tensor::ones([horizon, action_dim], (Kind::Float, device)) * 0.5;

        for _ in 0..self.config.mpc_iterations {
            // 1. Sample action sequences
            let actions = self.sample_actions(&mean, &std, samples);

            // 2. Evaluate sequences
            let scores = self.evaluate_actions(state, &actions);

            // 3. Update distribution based on elite candidates
            let elite_num = (samples as f32 * 0.1) as i64; // Top 10%
            let (_top_scores, top_idx) = scores.topk(elite_num, 0, true, true);

            // Shape: [EliteBatch, Horizon, ActionDim]
            let elite_actions = actions.index_select(0, &top_idx);

            // Update mean and std (clamped for stability)
            mean = elite_actions.mean_dim(Some(vec![0].as_slice()), false, Kind::Float);
            std = elite_actions
                .std_dim(Some(vec![0].as_slice()), false, false)
                .clamp(0.1, 2.0);
        }

        // Return first action from the mean (tanh squashed)
        mean.get(0).tanh()
    }

    /// Sample multiple action sequences from the current mean and std.
    fn sample_actions(&self, mean: &Tensor, std: &Tensor, samples: i64) -> Tensor {
        let (horizon, dim) = mean.size2().unwrap();
        let noise = Tensor::randn([samples, horizon, dim], (Kind::Float, mean.device()));
        // Actions are sampled in unbounded space then squashed by Tanh during evaluation/return
        mean.unsqueeze(0) + &noise * std.unsqueeze(0)
    }

    /// Evaluate action sequences by rolling out the world model.
    fn evaluate_actions(&self, state: &State, actions: &Tensor) -> Tensor {
        let (samples, horizon, _) = actions.size3().unwrap();

        // Pre-squash actions for simulation
        let squashed_actions = actions.tanh();

        // Prepare initial state batch by repeating the current state
        let deter = state.deter.repeat([samples, 1]);
        let stoch = state.stoch.repeat([samples, 1, 1]);
        let logits = state.logits.repeat([samples, 1, 1]);
        let mut state_batch = State {
            deter,
            stoch,
            logits,
        };

        let mut total_rewards = Tensor::zeros([samples], (Kind::Float, state.deter.device()));

        no_grad(|| {
            for t in 0..horizon {
                // Get actions for this timestep: [Samples, ActionDim]
                let action_t = squashed_actions.select(1, t);

                // Imagine next latent state
                state_batch = self.rssm.imagine(&action_t, &state_batch);

                // Predict reward
                let feat = state_batch.get_features();
                let reward = self.reward_head.forward(&feat).squeeze_dim(-1);
                total_rewards += reward;
            }
        });

        total_rewards
    }
}
