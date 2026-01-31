//! Abstract optimizer interface for PufferLib.

#[cfg(feature = "torch")]
use tch::{nn, Tensor};

/// Trait for back-end agnostic optimizers.
pub trait PuffOptimizer: Send {
    /// Zero out gradients.
    fn zero_grad(&mut self);

    /// Perform an optimization step.
    fn step(&mut self);

    #[cfg(feature = "torch")]
    /// Get the variables managed by this optimizer.
    fn variables(&self) -> &[Tensor] {
        &[]
    }

    #[cfg(feature = "torch")]
    /// Optional: Clip gradients for Torch-based optimizers.
    fn clip_grad_norm(&mut self, _max_norm: f64) {
        // Default implementation does nothing
    }
}

#[cfg(feature = "torch")]
/// Wrapper for Torch's nn::Optimizer.
pub struct TorchOptimizer {
    pub inner: nn::Optimizer,
    pub variables: Vec<Tensor>,
}

#[cfg(feature = "torch")]
impl TorchOptimizer {
    pub fn new(inner: nn::Optimizer, variables: Vec<Tensor>) -> Self {
        Self { inner, variables }
    }
}

#[cfg(feature = "torch")]
impl From<(nn::Optimizer, Vec<Tensor>)> for TorchOptimizer {
    fn from(tuple: (nn::Optimizer, Vec<Tensor>)) -> Self {
        Self::new(tuple.0, tuple.1)
    }
}

#[cfg(feature = "torch")]
impl PuffOptimizer for TorchOptimizer {
    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn step(&mut self) {
        self.inner.step();
    }

    fn variables(&self) -> &[Tensor] {
        &self.variables
    }

    fn clip_grad_norm(&mut self, max_norm: f64) {
        let mut global_norm = 0.0f64;
        for var in &self.variables {
            let grad = var.grad();
            if grad.defined() {
                global_norm += grad
                    .pow_tensor_scalar(2.0)
                    .sum(tch::Kind::Float)
                    .double_value(&[]);
            }
        }
        global_norm = global_norm.sqrt();

        if global_norm > max_norm {
            let clip_coef = max_norm / (global_norm + 1e-6);
            for var in &self.variables {
                let mut grad = var.grad();
                if grad.defined() {
                    let _ = grad.f_mul_scalar_(clip_coef);
                }
            }
        }
    }
}

#[cfg(feature = "torch")]
/// Helper for Automatic Mixed Precision (AMP) to prevent underflow.
pub struct GradScaler {
    scale: f64,
    growth_factor: f64,
    backoff_factor: f64,
    growth_interval: usize,
    growth_tracker: usize,
}

#[cfg(feature = "torch")]
impl GradScaler {
    pub fn new(initial_scale: f64) -> Self {
        Self {
            scale: initial_scale,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            growth_tracker: 0,
        }
    }

    /// Scale a loss tensor.
    pub fn scale(&self, loss: &Tensor) -> Tensor {
        loss * self.scale
    }

    /// Unscales the gradients of the provided variables.
    /// Returns true if no overflows were detected and the optimizer step should proceed.
    pub fn unscale(&mut self, variables: &[Tensor], max_norm: Option<f64>) -> bool {
        let inv_scale = Tensor::from(1.0 / self.scale);
        let mut found_inf = Tensor::from(0.0);

        for var in variables {
            let mut grad = var.grad();
            if grad.defined() {
                grad.internal_amp_non_finite_check_and_unscale(&mut found_inf, &inv_scale);
            }
        }

        let overflow = found_inf.double_value(&[]) > 0.0;

        if overflow {
            self.scale *= self.backoff_factor;
            self.growth_tracker = 0;
            false
        } else {
            if let Some(norm) = max_norm {
                // Clip gradients after unscaling
                let mut total_norm = 0.0f64;
                for var in variables {
                    let grad = var.grad();
                    if grad.defined() {
                        let n = grad.norm();
                        total_norm += n.double_value(&[]).powi(2);
                    }
                }
                total_norm = total_norm.sqrt();

                if total_norm > norm {
                    let clip_coef = norm / (total_norm + 1e-6);
                    for var in variables {
                        let mut grad = var.grad();
                        if grad.defined() {
                            let _ = grad.f_mul_scalar_(clip_coef);
                        }
                    }
                }
            }

            self.growth_tracker += 1;
            if self.growth_tracker >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.growth_tracker = 0;
            }
            true
        }
    }

    pub fn current_scale(&self) -> f64 {
        self.scale
    }
}
