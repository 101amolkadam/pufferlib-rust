//! LSTM policy wrapper.

use super::Policy;
use tch::{nn, nn::Module, nn::RNN, Device, Kind, Tensor};

/// LSTM policy that wraps another policy
pub struct LstmPolicy {
    /// Input size (from base policy encoder)
    _input_size: i64,
    /// Hidden size for LSTM
    hidden_size: i64,
    /// LSTM layer
    lstm: nn::LSTM,
    /// Variable store
    vs: nn::VarStore,
    /// Actor head
    actor: nn::Linear,
    /// Critic head
    critic: nn::Linear,
    /// Encoder from base policy
    encoder: nn::Sequential,
    /// Device
    device: Device,
}

impl LstmPolicy {
    /// Create a new LSTM policy
    pub fn new(obs_size: i64, num_actions: i64, hidden_size: i64, device: Device) -> Self {
        let vs = nn::VarStore::new(device);
        let root = vs.root();

        // Simple encoder
        let encoder = nn::seq()
            .add(nn::linear(
                &root / "encoder",
                obs_size,
                hidden_size,
                Default::default(),
            ))
            .add_fn(|x| x.relu());

        // LSTM
        let lstm = nn::lstm(&root / "lstm", hidden_size, hidden_size, Default::default());

        // Output heads
        let actor = nn::linear(
            &root / "actor",
            hidden_size,
            num_actions,
            Default::default(),
        );
        let critic = nn::linear(&root / "critic", hidden_size, 1, Default::default());

        Self {
            _input_size: hidden_size,
            hidden_size,
            lstm,
            vs,
            actor,
            critic,
            encoder,
            device,
        }
    }

    /// Forward with LSTM state
    pub fn forward_with_state(
        &self,
        observations: &Tensor,
        hidden_state: Option<(Tensor, Tensor)>,
    ) -> (super::Distribution, Tensor, (Tensor, Tensor)) {
        let obs = observations.to_device(self.device);
        let batch_size = obs.size()[0];

        // Encode observations
        let encoded = self.encoder.forward(&obs);

        // Reshape for LSTM: (seq_len, batch, features)
        let encoded = encoded.unsqueeze(0);

        // Initialize hidden state if not provided
        let (h0, c0) = hidden_state.unwrap_or_else(|| {
            let h = Tensor::zeros(
                [1, batch_size, self.hidden_size],
                (Kind::Float, self.device),
            );
            let c = Tensor::zeros(
                [1, batch_size, self.hidden_size],
                (Kind::Float, self.device),
            );
            (h, c)
        });

        // LSTM forward
        let lstm_state = nn::LSTMState((h0, c0));
        let (output, new_state) = self.lstm.seq_init(&encoded, &lstm_state);

        // Get last output
        let hidden = output.squeeze_dim(0);

        // Compute outputs
        let logits = self.actor.forward(&hidden);
        let value = self.critic.forward(&hidden).squeeze_dim(-1);

        let (new_h, new_c) = new_state.0;
        (
            super::Distribution::Categorical { logits },
            value,
            (new_h, new_c),
        )
    }

    /// Get variable store
    pub fn var_store(&self) -> &nn::VarStore {
        &self.vs
    }

    /// Get mutable variable store
    pub fn var_store_mut(&mut self) -> &mut nn::VarStore {
        &mut self.vs
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.vs
            .trainable_variables()
            .iter()
            .map(|t| t.numel())
            .sum()
    }
}

impl Policy for LstmPolicy {
    fn forward(
        &self,
        observations: &Tensor,
        state: &Option<Vec<Tensor>>,
    ) -> (super::Distribution, Tensor, Option<Vec<Tensor>>) {
        let hidden_state = if let Some(s) = state {
            if s.len() >= 2 {
                Some((s[0].shallow_clone(), s[1].shallow_clone()))
            } else {
                None
            }
        } else {
            None
        };

        let (dist, value, (new_h, new_c)) = self.forward_with_state(observations, hidden_state);
        (dist, value, Some(vec![new_h, new_c]))
    }

    fn initial_state(&self, batch_size: i64) -> Option<Vec<Tensor>> {
        let h = Tensor::zeros(
            [1, batch_size, self.hidden_size],
            (Kind::Float, self.device),
        );
        let c = Tensor::zeros(
            [1, batch_size, self.hidden_size],
            (Kind::Float, self.device),
        );
        Some(vec![h, c])
    }
}

impl super::HasVarStore for LstmPolicy {
    fn var_store_mut(&mut self) -> &mut nn::VarStore {
        &mut self.vs
    }

    fn var_store(&self) -> &nn::VarStore {
        &self.vs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lstm_creation() {
        let _policy = LstmPolicy::new(4, 2, 64, Device::Cpu);
    }

    #[test]
    fn test_lstm_forward() {
        let policy = LstmPolicy::new(4, 2, 64, Device::Cpu);
        let obs = Tensor::randn([8, 4], (Kind::Float, Device::Cpu));
        let (dist, value, state) = policy.forward(&obs, &None);

        match dist {
            super::super::Distribution::Categorical { logits } => {
                assert_eq!(logits.size(), [8, 2]);
            }
            _ => panic!("Expected categorical distribution"),
        }
        assert_eq!(value.size(), [8]);
        assert!(state.is_some());
    }

    #[test]
    fn test_lstm_with_state() {
        let policy = LstmPolicy::new(4, 2, 64, Device::Cpu);
        let obs = Tensor::randn([8, 4], (Kind::Float, Device::Cpu));

        // First forward (no state)
        let (_, _, state1) = policy.forward_with_state(&obs, None);

        // Second forward with previous state
        let (_, _, _state2) = policy.forward_with_state(&obs, Some(state1));
    }
}
