//! Multi-layer perceptron policy.

use super::Policy;
use crate::spaces::{DynSpace, Space};
use tch::{nn, nn::Module, Device, Tensor};

/// Configuration for MLP policy
#[derive(Clone, Debug)]
pub struct MlpConfig {
    /// Hidden layer size
    pub hidden_size: i64,
    /// Number of hidden layers
    pub num_layers: usize,
    /// Activation function (currently only ReLU)
    pub activation: Activation,
}

#[derive(Clone, Debug, Copy)]
pub enum Activation {
    ReLU,
    Tanh,
    Gelu,
}

impl Default for MlpConfig {
    fn default() -> Self {
        Self {
            hidden_size: 128,
            num_layers: 2,
            activation: Activation::ReLU,
        }
    }
}

/// Multi-layer perceptron policy
pub struct MlpPolicy {
    /// Variable store for parameters
    vs: nn::VarStore,
    /// Encoder network
    encoder: nn::Sequential,
    /// Actor head (action logits)
    actor: nn::Linear,
    /// Critic head (value estimate)
    critic: nn::Linear,
    /// Number of actions
    _num_actions: i64,
    /// Whether this is a continuous policy
    is_continuous: bool,
    /// Device
    device: Device,
}

impl MlpPolicy {
    /// Create a new MLP policy
    pub fn new(
        obs_size: i64,
        num_actions: i64,
        is_continuous: bool,
        config: MlpConfig,
        device: Device,
    ) -> Self {
        let vs = nn::VarStore::new(device);
        let root = vs.root();

        // Build encoder
        let mut encoder = nn::seq();
        let mut in_size = obs_size;

        for i in 0..config.num_layers {
            let layer = nn::linear(
                &root / format!("encoder_{}", i),
                in_size,
                config.hidden_size,
                Default::default(),
            );
            encoder = encoder.add(layer);

            match config.activation {
                Activation::ReLU => encoder = encoder.add_fn(|x| x.relu()),
                Activation::Tanh => encoder = encoder.add_fn(|x| x.tanh()),
                Activation::Gelu => encoder = encoder.add_fn(|x| x.gelu("none")),
            }

            in_size = config.hidden_size;
        }

        // Actor head
        let actor_out = if is_continuous {
            num_actions * 2 // Mean and log_std
        } else {
            num_actions
        };

        let actor = nn::linear(
            &root / "actor",
            config.hidden_size,
            actor_out,
            Default::default(),
        );
        let critic = nn::linear(&root / "critic", config.hidden_size, 1, Default::default());

        // Initialize weights
        Self::init_weights(&vs);

        Self {
            vs,
            encoder,
            actor,
            critic,
            _num_actions: num_actions,
            is_continuous,
            device,
        }
    }

    pub fn from_spaces(
        obs_space: &DynSpace,
        action_space: &DynSpace,
        config: MlpConfig,
        device: Device,
    ) -> Self {
        let obs_size = obs_space.shape().iter().product::<usize>() as i64;
        let (n_actions, is_cont) = match action_space {
            DynSpace::Discrete(d) => (d.n as i64, false),
            DynSpace::MultiDiscrete(md) => (md.nvec.iter().map(|&x| x as i64).sum(), false),
            DynSpace::Box(b) => (b.shape().iter().product::<usize>() as i64, true),
            _ => panic!("Unsupported action space for MlpPolicy"),
        };

        Self::new(obs_size, n_actions, is_cont, config, device)
    }

    /// Initialize weights with orthogonal initialization
    fn init_weights(vs: &nn::VarStore) {
        for (name, mut var) in vs.variables() {
            if name.contains("weight") {
                tch::no_grad(|| {
                    var.copy_(&(Tensor::randn_like(&var) * 0.01));
                });
            } else if name.contains("bias") {
                tch::no_grad(|| {
                    let _ = var.zero_();
                });
            }
        }
    }

    /// Get the variable store for optimization
    pub fn var_store(&self) -> &nn::VarStore {
        &self.vs
    }

    /// Get mutable variable store
    pub fn var_store_mut(&mut self) -> &mut nn::VarStore {
        &mut self.vs
    }

    /// Save policy to file
    pub fn save(&self, path: &str) -> Result<(), tch::TchError> {
        self.vs.save(path)
    }

    /// Load policy from file
    pub fn load(&mut self, path: &str) -> Result<(), tch::TchError> {
        self.vs.load(path)
    }

    /// Get the number of parameters
    pub fn num_parameters(&self) -> i64 {
        self.vs.variables().values().map(|v| v.numel() as i64).sum()
    }
}

impl Policy for MlpPolicy {
    fn forward(
        &self,
        observations: &Tensor,
        _state: &Option<Vec<Tensor>>,
    ) -> (super::Distribution, Tensor, Option<Vec<Tensor>>) {
        let obs = observations.to_device(self.device);
        let hidden = self.encoder.forward(&obs);
        let actor_out = self.actor.forward(&hidden);
        let value = self.critic.forward(&hidden).squeeze_dim(-1);

        let dist = if self.is_continuous {
            let mean_logstd = actor_out.chunk(2, -1);
            let mean = mean_logstd[0].shallow_clone();
            // log_std clamped for stability
            let log_std = mean_logstd[1].clamp(-20.0, 2.0);
            super::Distribution::Gaussian {
                mean,
                std: log_std.exp(),
            }
        } else {
            super::Distribution::Categorical { logits: actor_out }
        };

        (dist, value, None)
    }

    fn initial_state(&self, _batch_size: i64) -> Option<Vec<Tensor>> {
        None
    }
}

impl super::HasVarStore for MlpPolicy {
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
    use crate::spaces::Space;
    use tch::Kind;

    #[test]
    fn test_mlp_creation() {
        let _policy = MlpPolicy::new(4, 2, false, MlpConfig::default(), Device::Cpu);
    }

    #[test]
    fn test_mlp_forward() {
        let policy = MlpPolicy::new(4, 2, false, MlpConfig::default(), Device::Cpu);
        let obs = Tensor::randn([8, 4], (Kind::Float, Device::Cpu));
        let (dist, value, _) = policy.forward(&obs, &None);

        match dist {
            super::super::Distribution::Categorical { logits } => {
                assert_eq!(logits.size(), [8, 2]);
            }
            _ => panic!("Expected categorical distribution"),
        }
        assert_eq!(value.size(), [8]);
    }
}
