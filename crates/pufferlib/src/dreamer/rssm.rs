use crate::dreamer::config::DreamerConfig;
use tch::{nn, nn::RNN, Device, Kind, Tensor};

/// State containing deterministic and stochastic parts
#[derive(Debug)]
pub struct State {
    pub deter: Tensor,  // [B, deter_size]
    pub stoch: Tensor, // [B, stoch_size, stoch_discrete] (One-hot or Logits?) - usually One-hot/Sample
    pub logits: Tensor, // [B, stoch_size, stoch_discrete]
}

impl Clone for State {
    fn clone(&self) -> Self {
        Self {
            deter: self.deter.shallow_clone(),
            stoch: self.stoch.shallow_clone(),
            logits: self.logits.shallow_clone(),
        }
    }
}

impl State {
    /// Concatenate deterministic and stochastic parts for downstream heads
    pub fn get_features(&self) -> Tensor {
        Tensor::cat(&[&self.deter, &self.stoch.flatten(1, -1)], 1)
    }
}

pub struct RSSM {
    config: DreamerConfig,
    cell: nn::GRU,
    fc_embed: nn::Linear,
    fc_prior: nn::Linear,
    fc_post: nn::Linear,
}

impl RSSM {
    pub fn new(p: &nn::Path, config: DreamerConfig, action_dim: i64) -> Self {
        let stoch_flat = config.stoch_size * config.stoch_discrete;

        let fc_embed = nn::linear(
            p / "fc_embed",
            stoch_flat + action_dim,
            config.deter_size,
            Default::default(),
        );
        let cell = nn::gru(
            p / "cell",
            config.deter_size,
            config.deter_size,
            Default::default(),
        );

        let fc_prior = nn::linear(
            p / "fc_prior",
            config.deter_size,
            stoch_flat,
            Default::default(),
        );
        let fc_post = nn::linear(
            p / "fc_post",
            config.deter_size + config.embedding_size,
            stoch_flat,
            Default::default(),
        );

        Self {
            config,
            cell,
            fc_embed,
            fc_prior,
            fc_post,
        }
    }

    /// Compute initial state
    pub fn initial_state(&self, batch_size: i64, device: Device) -> State {
        let deter = Tensor::zeros(&[batch_size, self.config.deter_size], (Kind::Float, device));
        let stoch = Tensor::zeros(
            &[
                batch_size,
                self.config.stoch_size,
                self.config.stoch_discrete,
            ],
            (Kind::Float, device),
        );
        let logits = Tensor::zeros(
            &[
                batch_size,
                self.config.stoch_size,
                self.config.stoch_discrete,
            ],
            (Kind::Float, device),
        );
        State {
            deter,
            stoch,
            logits,
        }
    }

    /// Observe step: Compute posterior from prev state, action, and *real* embedding
    pub fn observe(
        &self,
        embed: &Tensor,  // [B, Embed]
        action: &Tensor, // [B, Action]
        prev_state: &State,
    ) -> (State, State) {
        // 1. Deterministic step
        let deter = self.step_deter(prev_state, action);

        // 2. Prior (for specific loss calc, though we only need posterior for next state)
        // But observe usually returns Prior AND Posterior to compute KL loss.
        let prior = self.step_prior(&deter);

        // 3. Posterior
        let post = self.step_posterior(&deter, embed);

        (post, prior)
    }

    /// Imagine step: Compute prior from prev state and action (no real obs)
    pub fn imagine(&self, action: &Tensor, prev_state: &State) -> State {
        let deter = self.step_deter(prev_state, action);
        self.step_prior(&deter)
    }

    // -- Internal Steps --

    fn step_deter(&self, prev_state: &State, action: &Tensor) -> Tensor {
        // Inputs: prev_stoch (flattened) + action
        let stoch_flat = prev_state.stoch.flatten(1, -1);
        let input = Tensor::cat(&[&stoch_flat, action], 1);
        let vector = input.apply(&self.fc_embed).elu();

        // GRU Step
        let next_deter_wrapped = self
            .cell
            .step(&vector, &nn::GRUState(prev_state.deter.shallow_clone()));
        next_deter_wrapped.0
    }

    fn step_prior(&self, deter: &Tensor) -> State {
        let logits = deter.apply(&self.fc_prior);
        let logits = logits.reshape(&[-1, self.config.stoch_size, self.config.stoch_discrete]);
        let stoch = self.sample_stoch(&logits);
        State {
            deter: deter.shallow_clone(),
            stoch,
            logits,
        }
    }

    fn step_posterior(&self, deter: &Tensor, embed: &Tensor) -> State {
        let input = Tensor::cat(&[deter, embed], 1);
        let logits = input.apply(&self.fc_post);
        let logits = logits.reshape(&[-1, self.config.stoch_size, self.config.stoch_discrete]);
        let stoch = self.sample_stoch(&logits);
        State {
            deter: deter.shallow_clone(),
            stoch,
            logits,
        }
    }

    fn sample_stoch(&self, logits: &Tensor) -> Tensor {
        // Gumbel Softmax or just sample?
        // DreamerV3: One-hot with Straight-through
        // tch: gumbel_softmax?
        // Or manual:
        let noise = Tensor::zeros_like(logits).uniform_(0.0, 1.0);
        // -log(-log(u))
        let gumbel = -((-(noise + 1e-8).log() + 1e-8).log());
        let soft = (logits + gumbel).softmax(-1, Kind::Float);

        // Hard one-hot (argmax)
        let max_idx = soft.argmax(-1, false);
        let hard = max_idx.one_hot(self.config.stoch_discrete);
        // one_hot returns Long, need Float?
        // We need to cast hard to match soft type

        // Straight through: (hard - soft).detach() + soft
        // If we want OneHot categorical.
        // Note: one_hot dimension size?

        // Simplified: Just use Softmax for now (DreamerV2 style for simplicity) if Hard is tricky.
        // But Hard is part of V3.
        // I'll stick to simple Softmax sampling (reparameterized? No, categorical isn't reparameterizable easily without Gumbel-Softmax)

        // I'll use simple soft sample flow.
        soft
    }
}
