//! Probability distributions for RL policies.

#[cfg(feature = "torch")]
use tch::{Kind, Tensor as TorchTensor};

#[cfg(feature = "candle")]
use candle_core::Tensor as CandleTensor;

#[cfg(feature = "burn")]
use burn_core::tensor::{backend::Backend as BurnBackend, Tensor as BurnTensor};
#[cfg(feature = "burn")]
pub type PufferBurnBackend = burn_ndarray::NdArray;

/// Enum for different types of probability distributions
pub enum Distribution {
    #[cfg(feature = "torch")]
    Categorical { logits: TorchTensor },
    #[cfg(feature = "torch")]
    Gaussian { mean: TorchTensor, std: TorchTensor },

    #[cfg(feature = "candle")]
    CandleCategorical { logits: CandleTensor },
    #[cfg(feature = "candle")]
    CandleGaussian {
        mean: CandleTensor,
        std: CandleTensor,
    },

    #[cfg(feature = "burn")]
    BurnCategorical {
        logits: BurnTensor<PufferBurnBackend, 2>,
    },
    #[cfg(feature = "burn")]
    BurnGaussian {
        mean: BurnTensor<PufferBurnBackend, 2>,
        std: BurnTensor<PufferBurnBackend, 2>,
    },
}

/// Helper to handle heterogeneous sample types
pub enum DistributionSample {
    #[cfg(feature = "torch")]
    Torch(TorchTensor),
    #[cfg(feature = "candle")]
    Candle(CandleTensor),
    #[cfg(feature = "burn")]
    Burn(BurnTensor<PufferBurnBackend, 2>), // Assuming 2D [batch, dim] for actions/samples
}

impl DistributionSample {
    #[cfg(feature = "torch")]
    pub fn as_torch(&self) -> &TorchTensor {
        match self {
            DistributionSample::Torch(t) => t,
            #[allow(unreachable_patterns)]
            _ => panic!("Not a torch tensor"),
        }
    }

    #[cfg(feature = "candle")]
    pub fn as_candle(&self) -> &CandleTensor {
        match self {
            DistributionSample::Candle(t) => t,
            #[allow(unreachable_patterns)]
            _ => panic!("Not a candle tensor"),
        }
    }

    #[cfg(feature = "burn")]
    pub fn as_burn(&self) -> &BurnTensor<PufferBurnBackend, 2> {
        match self {
            DistributionSample::Burn(t) => t,
            #[allow(unreachable_patterns)]
            _ => panic!("Not a burn tensor"),
        }
    }
}

impl Distribution {
    /// Sample raw actions from the distribution
    pub fn sample(&self) -> DistributionSample {
        match self {
            #[cfg(feature = "torch")]
            Distribution::Categorical { logits } => DistributionSample::Torch(
                logits
                    .softmax(-1, Kind::Float)
                    .multinomial(1, true)
                    .squeeze_dim(-1),
            ),
            #[cfg(feature = "torch")]
            Distribution::Gaussian { mean, std } => {
                let noise = TorchTensor::randn_like(mean);
                DistributionSample::Torch(mean + noise * std)
            }
            #[cfg(feature = "candle")]
            Distribution::CandleCategorical { logits } => {
                let probs = candle_nn::ops::softmax(logits, candle_core::D::Minus1).unwrap();
                DistributionSample::Candle(probs.argmax(candle_core::D::Minus1).unwrap())
            }
            #[cfg(feature = "candle")]
            Distribution::CandleGaussian { mean, std } => {
                let noise = CandleTensor::randn_like(mean, 0.0, 1.0).unwrap();
                DistributionSample::Candle((mean + (noise * std).unwrap()).unwrap())
            }
            #[cfg(feature = "burn")]
            Distribution::BurnCategorical { logits } => {
                let shape = logits.shape();
                let u = BurnTensor::<PufferBurnBackend, 2>::random(
                    shape.clone(),
                    burn_core::tensor::Distribution::Uniform(0.0, 1.0),
                    &logits.device(),
                );
                // Gumbel = -log(-log(u))
                let gumbel = u.log().mul_scalar(-1.0).log().mul_scalar(-1.0);
                let perturbed = logits.clone().add(gumbel);
                let sample_int = perturbed.argmax(1); // Tensor<B, 2, Int> (keeps dims usually in Burn?)
                                                      // If it reduces, we reshape. Burn's argmax reduces dimension in some versions, keeps in others.
                                                      // Let's assume it might reduce. But we want 2D [Batch, 1] for consistency?
                                                      // Actually `DistributionSample::Burn` is `BurnTensor<..., 2>`.
                                                      // If argmax returns [Batch], we need [Batch, 1].
                let sample_float = sample_int.float(); // Convert to float tensor
                                                       // Ensure 2D
                let dims = sample_float.dims();
                if dims.len() == 1 {
                    DistributionSample::Burn(sample_float.unsqueeze_dim(1))
                } else {
                    DistributionSample::Burn(sample_float)
                }
            }
            #[cfg(feature = "burn")]
            Distribution::BurnGaussian { mean, std } => {
                let shape = mean.shape();
                let noise = BurnTensor::<PufferBurnBackend, 2>::random(
                    shape,
                    burn_core::tensor::Distribution::Normal(0.0, 1.0),
                    &mean.device(),
                );
                DistributionSample::Burn(mean.clone().add(noise.mul(std.clone())))
            }
        }
    }

    /// Compute log probabilities for given actions
    pub fn log_prob(&self, actions: &DistributionSample) -> DistributionSample {
        match (self, actions) {
            #[cfg(feature = "torch")]
            (Distribution::Categorical { logits }, DistributionSample::Torch(ref actions)) => {
                let log_probs = logits.log_softmax(-1, Kind::Float);
                let indices = if actions.dim() == log_probs.dim() {
                    actions.to_kind(Kind::Int64)
                } else {
                    actions.unsqueeze(-1).to_kind(Kind::Int64)
                };
                DistributionSample::Torch(log_probs.gather(-1, &indices, false).squeeze_dim(-1))
            }
            #[cfg(feature = "torch")]
            (Distribution::Gaussian { mean, std }, DistributionSample::Torch(ref actions)) => {
                let var = std.pow_tensor_scalar(2.0);
                let log_std = std.log();

                let log_2pi = (2.0 * std::f64::consts::PI).ln();
                let log_2pi_tensor = TorchTensor::from(log_2pi).to_device(mean.device());
                let sq_diff = (actions - mean).pow_tensor_scalar(2.0);
                let element_wise_log_prob =
                    (sq_diff / (var + 1e-8) + log_std * 2.0 + log_2pi_tensor) * -0.5;
                DistributionSample::Torch(element_wise_log_prob.sum_dim_intlist(
                    [-1i64].as_slice(),
                    false,
                    Kind::Float,
                ))
            }
            #[cfg(feature = "candle")]
            (
                Distribution::CandleCategorical { logits },
                DistributionSample::Candle(ref actions),
            ) => {
                let log_probs = candle_nn::ops::log_softmax(logits, candle_core::D::Minus1)
                    .expect("log_softmax failed");
                // Gather log_probs based on actions
                // actions shape: [N], log_probs shape: [N, C]
                let actions_expanded = actions
                    .unsqueeze(candle_core::D::Minus1)
                    .expect("unsqueeze failed");
                DistributionSample::Candle(
                    log_probs
                        .gather(&actions_expanded, candle_core::D::Minus1)
                        .expect("gather failed")
                        .squeeze(candle_core::D::Minus1)
                        .expect("squeeze failed"),
                )
            }
            #[cfg(feature = "candle")]
            (
                Distribution::CandleGaussian { mean, std },
                DistributionSample::Candle(ref actions),
            ) => {
                let var = (std * std).expect("pow failed");
                let log_std = std.log().expect("log failed");
                let log_2pi = (2.0 * std::f64::consts::PI).ln();

                let sq_diff = ((actions - mean).expect("sub failed")
                    * (actions - mean).expect("sub failed"))
                .expect("mul failed");
                let element_wise_log_prob = ((sq_diff
                    .div(&(var + 1e-8).expect("add failed"))
                    .expect("div failed")
                    + (log_std * 2.0)
                        .expect("mul failed")
                        .add(&candle_core::Tensor::new(log_2pi, mean.device()).expect("new failed"))
                        .expect("add failed"))
                .expect("add failed")
                    * -0.5)
                    .expect("final mul failed");

                DistributionSample::Candle(
                    element_wise_log_prob
                        .sum(candle_core::D::Minus1)
                        .expect("sum failed"),
                )
            }
            #[cfg(feature = "burn")]
            (Distribution::BurnCategorical { logits }, DistributionSample::Burn(actions)) => {
                let log_probs = burn_core::tensor::activation::log_softmax(logits.clone(), 1);
                // actions is float (from sample), convert to int for gather
                let indices = actions.int();
                let gathered = log_probs.gather(1, indices);
                DistributionSample::Burn(gathered)
            }
            #[cfg(feature = "burn")]
            (Distribution::BurnGaussian { mean, std }, DistributionSample::Burn(actions)) => {
                let var = std.clone().powf_scalar(2.0);
                let log_std = std.clone().log();
                let log_2pi = (2.0 * std::f64::consts::PI).ln();

                let diff = actions.clone().sub(mean.clone());
                let sq_diff = diff.powf_scalar(2.0);

                // -0.5 * (sq_diff / (var + eps) + 2*log_std + log_2pi)
                let term1 = sq_diff.div(var.add_scalar(1e-8));
                let term2 = log_std.mul_scalar(2.0);
                let term3 = term1.add(term2).add_scalar(log_2pi);
                let log_prob = term3.mul_scalar(-0.5);

                // Sum along last dim (actions)
                DistributionSample::Burn(log_prob.sum_dim(1))
            }
            #[allow(unreachable_patterns)]
            _ => panic!("Backend mismatch in log_prob"),
        }
    }

    /// Compute entropy of the distribution
    pub fn entropy(&self) -> DistributionSample {
        match self {
            #[cfg(feature = "torch")]
            Self::Categorical { logits } => {
                let probs = logits.softmax(-1, Kind::Float);
                let log_probs = logits.log_softmax(-1, Kind::Float);
                let entropy =
                    -(probs * log_probs).sum_dim_intlist(Some(&[-1_i64][..]), false, Kind::Float);
                DistributionSample::Torch(entropy)
            }
            #[cfg(feature = "torch")]
            Self::Gaussian { mean: _, std } => {
                let entropy = std.log() + 0.5 + 0.5 * (2.0 * std::f64::consts::PI).ln();
                DistributionSample::Torch(entropy.sum_dim_intlist(
                    Some(&[-1_i64][..]),
                    false,
                    Kind::Float,
                ))
            }
            #[cfg(feature = "candle")]
            Self::CandleCategorical { logits } => {
                let probs = candle_nn::ops::softmax(logits, candle_core::D::Minus1)
                    .expect("softmax failed");
                let log_probs = candle_nn::ops::log_softmax(logits, candle_core::D::Minus1)
                    .expect("log_softmax failed");
                DistributionSample::Candle(
                    ((&probs * &log_probs)
                        .expect("mul failed")
                        .sum(candle_core::D::Minus1)
                        .expect("sum failed")
                        * -1.0)
                        .expect("negation failed"),
                )
            }
            #[cfg(feature = "candle")]
            Self::CandleGaussian { mean: _, std } => {
                // H = log(std * sqrt(2 * pi * e))
                let log_std = std.log().expect("log failed");
                let entropy = (log_std + (0.5 + 0.5 * (2.0 * std::f64::consts::PI).ln()))
                    .expect("entropy computation failed");
                DistributionSample::Candle(entropy.sum(candle_core::D::Minus1).expect("sum failed"))
            }
            #[cfg(feature = "burn")]
            Self::BurnCategorical { logits } => {
                let probs = burn_core::tensor::activation::softmax(logits.clone(), 1);
                let log_probs = burn_core::tensor::activation::log_softmax(logits.clone(), 1);
                let entropy = probs.mul(log_probs).sum_dim(1).mul_scalar(-1.0);
                DistributionSample::Burn(entropy)
            }
            #[cfg(feature = "burn")]
            Self::BurnGaussian { mean: _, std } => {
                let log_std = std.clone().log();
                let constant = 0.5 + 0.5 * (2.0 * std::f64::consts::PI).ln();
                let entropy = log_std.add_scalar(constant);
                DistributionSample::Burn(entropy.sum_dim(1))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "torch")]
    use tch::{Device, Kind, Tensor};

    #[test]
    #[cfg(feature = "torch")]
    fn test_categorical_sample() {
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 10.0]).reshape([1, 3]);
        let dist = Distribution::Categorical { logits };
        let sample = dist.sample();
        let val = sample.as_torch().get(0).double_value(&[]);
        assert!(val >= 0.0 && val <= 2.0);
    }

    #[test]
    #[cfg(feature = "torch")]
    fn test_gaussian_sample() {
        let mean = Tensor::zeros([1, 2], (Kind::Float, Device::Cpu));
        let std = Tensor::from_slice(&[1.0f32, 2.0]).reshape([1, 2]);
        let dist = Distribution::Gaussian { mean, std };
        let sample = dist.sample();
        assert_eq!(sample.as_torch().size(), [1, 2]);
    }

    #[test]
    #[cfg(feature = "torch")]
    fn test_gaussian_log_prob() {
        let mean = Tensor::zeros([1, 1], (Kind::Float, Device::Cpu));
        let std = Tensor::ones([1, 1], (Kind::Float, Device::Cpu));
        let dist = Distribution::Gaussian { mean, std };
        let x = Tensor::zeros([1, 1], (Kind::Float, Device::Cpu));
        let x_sample = DistributionSample::Torch(x);
        let log_prob = dist.log_prob(&x_sample);
        let val = log_prob.as_torch().get(0).double_value(&[]);
        assert!((val + 0.9189).abs() < 1e-4);
    }

    #[test]
    #[cfg(feature = "torch")]
    fn test_gaussian_entropy() {
        let mean = Tensor::zeros([1, 1], (Kind::Float, Device::Cpu));
        let std = Tensor::ones([1, 1], (Kind::Float, Device::Cpu));
        let dist = Distribution::Gaussian { mean, std };
        let entropy = dist.entropy();
        let val = entropy.as_torch().get(0).double_value(&[]);
        assert!((val - 1.4189).abs() < 1e-4);
    }
}
