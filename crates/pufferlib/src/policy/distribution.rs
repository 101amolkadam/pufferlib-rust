//! Probability distributions for RL policies.

use tch::{Kind, Tensor};

/// Enum for different types of probability distributions
pub enum Distribution {
    /// Categorical distribution for discrete action spaces
    Categorical { logits: Tensor },
    /// Gaussian distribution for continuous action spaces
    Gaussian { mean: Tensor, std: Tensor },
}

impl Distribution {
    /// Sample raw actions from the distribution
    pub fn sample(&self) -> Tensor {
        match self {
            Distribution::Categorical { logits } => logits
                .softmax(-1, Kind::Float)
                .multinomial(1, true)
                .squeeze_dim(-1),
            Distribution::Gaussian { mean, std } => {
                let noise = Tensor::randn_like(mean);
                mean + noise * std
            }
        }
    }

    /// Compute log probabilities for given actions
    pub fn log_prob(&self, actions: &Tensor) -> Tensor {
        match self {
            Distribution::Categorical { logits } => {
                let log_probs = logits.log_softmax(-1, Kind::Float);
                log_probs
                    .gather(-1, &actions.unsqueeze(-1).to_kind(Kind::Int64), false)
                    .squeeze_dim(-1)
            }
            Distribution::Gaussian { mean, std } => {
                let var = std.pow_tensor_scalar(2.0);
                let log_std = std.log();

                let log_2pi = (2.0 * std::f64::consts::PI).ln();
                let log_2pi_tensor = Tensor::from(log_2pi).to_device(mean.device());
                let sq_diff = (actions - mean).pow_tensor_scalar(2.0);
                let element_wise_log_prob =
                    (sq_diff / (var + 1e-8) + log_std * 2.0 + log_2pi_tensor) * -0.5;
                element_wise_log_prob.sum_dim_intlist([-1i64].as_slice(), false, Kind::Float)
            }
        }
    }

    /// Compute entropy of the distribution
    pub fn entropy(&self) -> Tensor {
        match self {
            Distribution::Categorical { logits } => {
                let probs = logits.softmax(-1, Kind::Float);
                let log_probs = logits.log_softmax(-1, Kind::Float);
                -(probs * log_probs).sum_dim_intlist([-1i64].as_slice(), false, Kind::Float)
            }
            Distribution::Gaussian { std, .. } => {
                let log_std = std.log();
                let log_2pi_e = (2.0 * std::f32::consts::PI * std::f32::consts::E).ln() as f64;
                (log_std + 0.5 * log_2pi_e).sum_dim_intlist([-1i64].as_slice(), false, Kind::Float)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_categorical_sample() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 10.0]).reshape([1, 3]);
        let dist = Distribution::Categorical { logits };
        let sample = dist.sample();
        assert_eq!(sample.size(), [1]);
        let val = sample.get(0).double_value(&[]);
        assert!(val >= 0.0 && val <= 2.0);
    }

    #[test]
    fn test_gaussian_sample() {
        let mean = Tensor::zeros([1, 2], (Kind::Float, Device::Cpu));
        let std = Tensor::from_slice(&[1.0f32, 2.0]).reshape([1, 2]);
        let dist = Distribution::Gaussian { mean, std };
        let sample = dist.sample();
        assert_eq!(sample.size(), [1, 2]);
    }

    #[test]
    fn test_gaussian_log_prob() {
        // For mean=0, std=1, x=0: log_prob = -0.5 * (0 + 0 + log(2*pi)) = -0.5 * 1.8378 = -0.9189
        let mean = Tensor::zeros([1, 1], (Kind::Float, Device::Cpu));
        let std = Tensor::ones([1, 1], (Kind::Float, Device::Cpu));
        let dist = Distribution::Gaussian { mean, std };
        let x = Tensor::zeros([1, 1], (Kind::Float, Device::Cpu));
        let log_prob = dist.log_prob(&x);
        let val = log_prob.get(0).double_value(&[]);
        assert!((val + 0.9189).abs() < 1e-4);
    }

    #[test]
    fn test_gaussian_entropy() {
        // For std=1: entropy = 0.5 * log(2 * pi * e) = 0.5 * 2.8378 = 1.4189
        let mean = Tensor::zeros([1, 1], (Kind::Float, Device::Cpu));
        let std = Tensor::ones([1, 1], (Kind::Float, Device::Cpu));
        let dist = Distribution::Gaussian { mean, std };
        let entropy = dist.entropy();
        let val = entropy.get(0).double_value(&[]);
        assert!((val - 1.4189).abs() < 1e-4);
    }
}
