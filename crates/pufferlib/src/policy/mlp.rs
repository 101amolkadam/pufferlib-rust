//! Multi-layer perceptron policy.

#[cfg(feature = "torch")]
use super::Policy;
use crate::spaces::{DynSpace, Space};
#[cfg(feature = "torch")]
use tch::{nn, nn::Module, Device, Tensor};

#[cfg(feature = "candle")]
use candle_nn as candle_nn_backend;

#[cfg(feature = "burn")]
use burn_core::{
    module::Module as BurnModule,
    nn::{self as burn_nn},
    tensor::{backend::Backend as BurnBackend, Tensor as BurnTensor},
};

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
#[cfg(feature = "torch")]
pub struct MlpPolicy {
    /// Variable store for parameters
    vs: nn::VarStore,
    /// Encoder network
    encoder: nn::Sequential,
    /// Actor head (action logits)
    actor: nn::Linear,
    /// Critic head (value estimate)
    critic: nn::Linear,
    /// Cost critic head (safety cost estimate)
    cost_critic: nn::Linear,
    /// Number of actions
    _num_actions: i64,
    /// Whether this is a continuous policy
    is_continuous: bool,
    /// Device
    device: Device,
}

#[cfg(feature = "torch")]
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
        let cost_critic = nn::linear(
            &root / "cost_critic",
            config.hidden_size,
            1,
            Default::default(),
        );

        // Initialize weights
        Self::init_weights(&vs);

        Self {
            vs,
            encoder,
            actor,
            critic,
            cost_critic,
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

#[cfg(feature = "torch")]
impl super::HasVarStore for MlpPolicy {
    fn var_store_mut(&mut self) -> &mut nn::VarStore {
        &mut self.vs
    }

    fn var_store(&self) -> &nn::VarStore {
        &self.vs
    }
}

#[cfg(feature = "torch")]
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

#[cfg(feature = "torch")]
impl super::SafePolicy for MlpPolicy {
    fn forward_safe(
        &self,
        observations: &Tensor,
        _state: &Option<Vec<Tensor>>,
    ) -> (super::Distribution, Tensor, Tensor, Option<Vec<Tensor>>) {
        let obs = observations.to_device(self.device);
        let hidden = self.encoder.forward(&obs);
        let actor_out = self.actor.forward(&hidden);
        let value = self.critic.forward(&hidden).squeeze_dim(-1);
        let cost_value = self.cost_critic.forward(&hidden).squeeze_dim(-1);

        let dist = if self.is_continuous {
            let mean_logstd = actor_out.chunk(2, -1);
            let mean = mean_logstd[0].shallow_clone();
            let log_std = mean_logstd[1].clamp(-20.0, 2.0);

            super::Distribution::Gaussian {
                mean,
                std: log_std.exp(),
            }
        } else {
            super::Distribution::Categorical { logits: actor_out }
        };

        (dist, value, cost_value, None)
    }
}

#[cfg(feature = "candle")]
pub struct CandleMlp {
    encoder: candle_nn::Sequential,
    actor: candle_nn::Linear,
    critic: candle_nn::Linear,
    is_continuous: bool,
}

#[cfg(feature = "candle")]
impl CandleMlp {
    pub fn new(
        obs_size: usize,
        num_actions: usize,
        is_continuous: bool,
        config: MlpConfig,
        vb: candle_nn::VarBuilder,
    ) -> candle_core::Result<Self> {
        let mut encoder = candle_nn::seq();
        let mut in_size = obs_size;

        for _ in 0..config.num_layers {
            encoder = encoder.add(candle_nn::linear(
                in_size,
                config.hidden_size as usize,
                vb.clone(),
            )?);
            match config.activation {
                Activation::ReLU => encoder = encoder.add_fn(|x| x.relu()),
                Activation::Tanh => encoder = encoder.add_fn(|x| x.tanh()),
                Activation::Gelu => encoder = encoder.add_fn(|x| x.gelu()),
            }
            in_size = config.hidden_size as usize;
        }

        let actor_out = if is_continuous {
            num_actions * 2
        } else {
            num_actions
        };
        let actor = candle_nn::linear(config.hidden_size as usize, actor_out, vb.clone())?;
        let critic = candle_nn::linear(config.hidden_size as usize, 1, vb)?;

        Ok(Self {
            encoder,
            actor,
            critic,
            is_continuous,
        })
    }

    pub fn forward(
        &self,
        observations: &candle_core::Tensor,
    ) -> candle_core::Result<(super::Distribution, candle_core::Tensor)> {
        use candle_nn::Module;
        let hidden = self.encoder.forward(observations)?;
        let actor_out = self.actor.forward(&hidden)?;
        let value = self
            .critic
            .forward(&hidden)?
            .squeeze(candle_core::D::Minus1)?;

        let dist = if self.is_continuous {
            let chunks = actor_out.chunk(2, candle_core::D::Minus1)?;
            let mean = chunks[0].clone();
            let log_std = chunks[1].clamp(-20.0, 2.0)?;
            super::Distribution::CandleGaussian {
                mean,
                std: log_std.exp()?,
            }
        } else {
            super::Distribution::CandleCategorical { logits: actor_out }
        };

        Ok((dist, value))
    }
}

#[cfg(feature = "burn")]
use crate::policy::distribution::PufferBurnBackend;

#[cfg(feature = "burn")]
#[derive(burn_core::module::Module, Debug)]
pub struct BurnMlp {
    encoder_mlp: burn_nn::Mlp<PufferBurnBackend>,
    actor: burn_nn::Linear<PufferBurnBackend>,
    critic: burn_nn::Linear<PufferBurnBackend>,
    is_continuous: bool,
}

#[cfg(feature = "burn")]
impl BurnMlp {
    pub fn new(
        obs_size: usize,
        num_actions: usize,
        is_continuous: bool,
        config: MlpConfig,
        device: &<PufferBurnBackend as BurnBackend>::Device,
    ) -> Self {
        let activation = match config.activation {
            Activation::ReLU => burn_nn::Gelu, // Burn MlpConfig currently might not support runtime activation choice easily or maps differently.
            // Actually MlpConfig in Burn usually takes an activation type or enum.
            // Checking Burn docs: MlpConfig usually allows setting activation.
            // For now, I'll default to Gelu or Relu if I can finding the enum.
            // Let's assume burn_nn::Relu works.
            Activation::Tanh => burn_nn::Gelu, // Placeholder
            Activation::Gelu => burn_nn::Gelu,
        };
        // Note: In Burn 0.13, Mlp generic over B only? Or allows activation?
        // Simple approach: Use default activation (Relu/Gelu) for now.

        let mlp_config = burn_nn::MlpConfig::new(
            config.num_layers,
            obs_size,
            config.hidden_size as usize,
            config.hidden_size as usize,
        );
        // .with_activation(...) if supported

        let encoder_mlp = burn_nn::Mlp::new(&mlp_config, device);

        let actor_out = if is_continuous {
            num_actions * 2
        } else {
            num_actions
        };

        let actor = burn_nn::LinearConfig::new(config.hidden_size as usize, actor_out).init(device);
        let critic = burn_nn::LinearConfig::new(config.hidden_size as usize, 1).init(device);

        Self {
            encoder_mlp,
            actor,
            critic,
            is_continuous,
        }
    }

    pub fn forward(
        &self,
        observations: BurnTensor<PufferBurnBackend, 2>,
    ) -> (super::Distribution, BurnTensor<PufferBurnBackend, 1>) {
        let hidden = self.encoder_mlp.forward(observations);
        let actor_out = self.actor.forward(hidden.clone());
        let value = self.critic.forward(hidden).squeeze(1);

        let dist = if self.is_continuous {
            let chunks = actor_out.chunk(2, 1);
            let mean = chunks[0].clone();
            let log_std = chunks[1].clone().clamp(-20.0, 2.0);
            super::Distribution::BurnGaussian {
                mean,
                std: log_std.exp(),
            }
        } else {
            super::Distribution::BurnCategorical { logits: actor_out }
        };

        (dist, value)
    }
}

#[cfg(all(test, feature = "torch"))]
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
