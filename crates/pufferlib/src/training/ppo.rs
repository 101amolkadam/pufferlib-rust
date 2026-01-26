//! PPO algorithm utilities.

use tch::{Kind, Tensor};

/// Compute Generalized Advantage Estimation (GAE)
///
/// # Arguments
/// * `rewards` - Tensor of rewards [T, N]
/// * `values` - Tensor of value estimates [T, N]
/// * `dones` - Tensor of done flags [T, N]
/// * `last_value` - Value estimate for terminal state [N]
/// * `gamma` - Discount factor
/// * `gae_lambda` - GAE lambda
///
/// # Returns
/// Advantages tensor [T, N]
pub fn compute_gae(
    rewards: &Tensor,
    values: &Tensor,
    dones: &Tensor,
    last_value: &Tensor,
    gamma: f64,
    gae_lambda: f64,
) -> Tensor {
    let device = rewards.device();
    let size = rewards.size();
    let steps = size[0] as usize;
    let num_envs = size[1] as usize;

    let advantages = Tensor::zeros(&size, (Kind::Float, device));
    let mut last_gae = Tensor::zeros([num_envs as i64], (Kind::Float, device));

    for t in (0..steps).rev() {
        let next_values = if t == steps - 1 {
            last_value.shallow_clone()
        } else {
            values.select(0, (t + 1) as i64).shallow_clone()
        };

        let r = rewards.select(0, t as i64);
        let d = dones.select(0, t as i64);
        let v = values.select(0, t as i64);

        let delta = &r + gamma * &next_values * (1.0 - &d) - &v;
        last_gae = &delta + gamma * gae_lambda * (1.0 - &d) * &last_gae;

        advantages.select(0, t as i64).copy_(&last_gae);
    }

    advantages
}

/// Compute V-trace corrected advantages
///
/// # Arguments
/// * `rewards` - Tensor of rewards [T, N]
/// * `values` - Tensor of value estimates [T, N]
/// * `dones` - Tensor of done flags [T, N]
/// * `importance` - Importance sampling ratios [T, N]
/// * `last_value` - Value estimate for terminal state [N]
/// * `gamma` - Discount factor
/// * `gae_lambda` - GAE lambda
/// * `rho_clip` - Clipping for rho (truncation)
/// * `c_clip` - Clipping for c (trace cutting)
///
/// # Returns
/// V-trace corrected advantages [T, N]
#[allow(clippy::too_many_arguments)]
pub fn compute_vtrace(
    rewards: &Tensor,
    values: &Tensor,
    dones: &Tensor,
    importance: &Tensor,
    last_value: &Tensor,
    gamma: f64,
    gae_lambda: f64,
    rho_clip: f64,
    c_clip: f64,
) -> Tensor {
    let device = rewards.device();
    let size = rewards.size();
    let steps = size[0] as usize;
    let num_envs = size[1] as usize;

    // Clip importance weights
    let rho = importance.clamp_max(rho_clip);
    let c = importance.clamp_max(c_clip);

    let advantages = Tensor::zeros(&size, (Kind::Float, device));
    let mut last_gae = Tensor::zeros([num_envs as i64], (Kind::Float, device));

    for t in (0..steps).rev() {
        let next_values = if t == steps - 1 {
            last_value.shallow_clone()
        } else {
            values.select(0, (t + 1) as i64).shallow_clone()
        };

        let r = rewards.select(0, t as i64);
        let d = dones.select(0, t as i64);
        let v = values.select(0, t as i64);
        let rho_t = rho.select(0, t as i64);
        let c_t = c.select(0, t as i64);

        // V-trace delta with importance sampling
        let delta = &rho_t * (&r + gamma * &next_values * (1.0 - &d) - &v);
        last_gae = &delta + gamma * gae_lambda * &c_t * (1.0 - &d) * &last_gae;

        advantages.select(0, t as i64).copy_(&last_gae);
    }

    advantages
}

/// Compute PPO clipped policy loss
pub fn ppo_policy_loss(
    advantages: &Tensor,
    log_probs: &Tensor,
    old_log_probs: &Tensor,
    clip_coef: f64,
) -> Tensor {
    let ratio = (log_probs - old_log_probs).exp();

    let surr1 = &ratio * advantages;
    let surr2 = ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef) * advantages;

    -surr1.min_other(&surr2).mean(Kind::Float)
}

/// Compute clipped value loss
pub fn ppo_value_loss(
    values: &Tensor,
    old_values: &Tensor,
    returns: &Tensor,
    clip_coef: f64,
) -> Tensor {
    let values_clipped = old_values + (values - old_values).clamp(-clip_coef, clip_coef);

    let loss1 = (values - returns).pow_tensor_scalar(2);
    let loss2 = (&values_clipped - returns).pow_tensor_scalar(2);

    loss1.max_other(&loss2).mean(Kind::Float) * 0.5
}

/// Compute KL divergence between old and new log probabilities
pub fn kl_divergence(log_probs: &Tensor, old_log_probs: &Tensor) -> Tensor {
    (old_log_probs - log_probs).mean(Kind::Float)
}

/// Compute PPO dual-clipped policy loss
/// Useful for continuous action spaces to prevent large updates
pub fn ppo_dual_clip_policy_loss(
    advantages: &Tensor,
    log_probs: &Tensor,
    old_log_probs: &Tensor,
    clip_coef: f64,
    dual_clip_coef: f64,
) -> Tensor {
    let ratio = (log_probs - old_log_probs).exp();

    let surr1 = &ratio * advantages;
    let surr2 = ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef) * advantages;

    let ppo_loss = surr1.min_other(&surr2);

    // Dual clip: for negative advantages, we don't want the ratios to be too small
    let is_neg_adv = advantages.lt(0.0).to_kind(Kind::Float);
    let dual_clipped = dual_clip_coef * advantages;

    // loss = max(ppo_loss, dual_clip * adv) for neg advantages
    let final_loss: Tensor =
        (1.0f64 - &is_neg_adv) * &ppo_loss + &is_neg_adv * ppo_loss.max_other(&dual_clipped);

    -final_loss.mean(Kind::Float)
}

/// Compute Soft Actor-Critic (SAC) style entropy-regularized loss
/// Note: Full SAC requires off-policy training, but this provides the focal point
/// for entropy maximization in on-policy settings.
#[allow(dead_code)]
pub fn sac_loss(values: &Tensor, log_probs: &Tensor, alpha: f64) -> Tensor {
    // SAC objective: J = E[Q(s,a) - alpha * log_p(a|s)]
    // In on-policy contexts, this often manifests as entropy maximization
    (alpha * log_probs - values).mean(Kind::Float)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_gae_computation() {
        let rewards = Tensor::ones([4, 2], (Kind::Float, Device::Cpu));
        let values = Tensor::zeros([4, 2], (Kind::Float, Device::Cpu));
        let dones = Tensor::zeros([4, 2], (Kind::Float, Device::Cpu));
        let last_value = Tensor::zeros([2], (Kind::Float, Device::Cpu));

        let advantages = compute_gae(&rewards, &values, &dones, &last_value, 0.99, 0.95);

        assert_eq!(advantages.size(), [4, 2]);
        // With all rewards = 1, values = 0, dones = 0, advantages should be positive
        assert!(advantages.mean(Kind::Float).double_value(&[]) > 0.0);
    }

    #[test]
    fn test_gae_with_dones() {
        // 5 steps, 1 env
        let rewards = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0, 1.0]).reshape([5, 1]);
        let values = Tensor::zeros([5, 1], (Kind::Float, Device::Cpu));
        // env resets at t=2
        let dones = Tensor::from_slice(&[0.0f32, 0.0, 1.0, 0.0, 0.0]).reshape([5, 1]);
        let last_value = Tensor::zeros([1], (Kind::Float, Device::Cpu));

        let advantages = compute_gae(&rewards, &values, &dones, &last_value, 0.9, 0.5);

        let adv_data = Vec::<f32>::try_from(advantages.flatten(0, -1)).unwrap();
        assert!(adv_data[1] >= 1.0);
        assert!(adv_data[2] >= 1.0);
    }

    #[test]
    fn test_kl_divergence() {
        let old_log_probs = Tensor::from_slice(&[-0.5f32, -1.0]).reshape([1, 2]);
        let log_probs = Tensor::from_slice(&[-0.6f32, -1.1]).reshape([1, 2]);
        let kl = kl_divergence(&log_probs, &old_log_probs);
        let val = kl.double_value(&[]);
        assert!((val - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_sac_loss() {
        let values = Tensor::from_slice(&[1.0f32, 2.0]).reshape([2]);
        let log_probs = Tensor::from_slice(&[-0.5f32, -1.0]).reshape([2]);
        let alpha = 0.5;

        let loss = sac_loss(&values, &log_probs, alpha);
        let val = loss.double_value(&[]);

        // loss = ((0.5 * -0.5 - 1.0) + (0.5 * -1.0 - 2.0)) / 2
        // loss = (-1.25 + -2.5) / 2 = -1.875
        assert!((val + 1.875).abs() < 1e-6);
    }
}
