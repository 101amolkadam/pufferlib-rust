use super::Policy;
use tch::{nn, nn::Module, Device, Tensor};

/// Convolutional Neural Network Policy
///
/// Standard nature-cnn style architecture:
/// 3 Convolutional layers -> Flatten -> Linear layers
pub struct CnnPolicy {
    /// Convolutional network
    features: nn::Sequential,
    /// Linear layer for actor (logits)
    actor: nn::Linear,
    /// Linear layer for critic (value)
    critic: nn::Linear,
    /// Device
    device: Device,
    /// Feature size after convolution
    _feature_size: i64,
}

impl CnnPolicy {
    /// Create a new CNN policy
    ///
    /// # Arguments
    /// * `input_channels` - Number of input channels (e.g. 3 for RGB)
    /// * `num_actions` - Number of output actions
    /// * `config` - Configuration (optional)
    /// * `device` - Device
    pub fn new(
        input_channels: i64,
        input_height: i64,
        input_width: i64,
        num_actions: i64,
        _config: super::mlp::MlpConfig, // Accept generic config for consistency
        var_store: &nn::Path,           // Use Path instead of VarStore ref specifically
    ) -> Self {
        // Nature-CNN architecture
        let conv1 = nn::conv2d(
            var_store / "c1",
            input_channels,
            32,
            8,
            nn::ConvConfig {
                stride: 4,
                ..Default::default()
            },
        );
        let conv2 = nn::conv2d(
            var_store / "c2",
            32,
            64,
            4,
            nn::ConvConfig {
                stride: 2,
                ..Default::default()
            },
        );
        let conv3 = nn::conv2d(
            var_store / "c3",
            64,
            64,
            3,
            nn::ConvConfig {
                stride: 1,
                ..Default::default()
            },
        );

        // Calculate output size dynamically
        let calc_conv = |size, kernel, stride| (size - kernel) / stride + 1;
        let h1 = calc_conv(input_height, 8, 4);
        let w1 = calc_conv(input_width, 8, 4);
        let h2 = calc_conv(h1, 4, 2);
        let w2 = calc_conv(w1, 4, 2);
        let h3 = calc_conv(h2, 3, 1);
        let w3 = calc_conv(w2, 3, 1);

        let features_dim = 64 * h3 * w3;
        let hidden_size = 512;

        let linear = nn::linear(
            var_store / "fc1",
            features_dim,
            hidden_size,
            Default::default(),
        );
        let actor = nn::linear(
            var_store / "actor",
            hidden_size,
            num_actions,
            Default::default(),
        );
        let critic = nn::linear(var_store / "critic", hidden_size, 1, Default::default());

        let features = nn::seq()
            .add(conv1)
            .add_fn(|x| x.relu())
            .add(conv2)
            .add_fn(|x| x.relu())
            .add(conv3)
            .add_fn(|x| x.relu())
            .add_fn(|x| x.flatten(1, -1))
            .add(linear)
            .add_fn(|x| x.relu());

        Self {
            features,
            actor,
            critic,
            device: Device::Cpu, // Default, updated on forward if needed? Or stored?
            _feature_size: hidden_size,
        }
    }

    pub fn num_parameters(&self) -> usize {
        // Approximate or implement properly using var_store scan
        0 // Placeholder
    }
}

impl Policy for CnnPolicy {
    fn forward(
        &self,
        observations: &Tensor,
        _state: &Option<Vec<Tensor>>,
    ) -> (super::Distribution, Tensor, Option<Vec<Tensor>>) {
        // Preprocess: assume NCHW or NHWC? PufferLib usually standardizes.
        // Assume obs are [Batch, Channels, Height, Width] normalized float
        let x = observations.to_device(self.device);
        let feats = self.features.forward(&x);

        let logits = self.actor.forward(&feats);
        let value = self.critic.forward(&feats).squeeze_dim(-1);

        (super::Distribution::Categorical { logits }, value, None)
    }

    fn initial_state(&self, _batch_size: i64) -> Option<Vec<Tensor>> {
        None
    }
}
