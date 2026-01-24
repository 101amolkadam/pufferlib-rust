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
        
        let _ = advantages.select(0, t as i64).copy_(&last_gae);
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
        
        let _ = advantages.select(0, t as i64).copy_(&last_gae);
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gae_computation() {
        use tch::Device;
        let rewards = Tensor::ones([4, 2], (Kind::Float, Device::Cpu));
        let values = Tensor::zeros([4, 2], (Kind::Float, Device::Cpu));
        let dones = Tensor::zeros([4, 2], (Kind::Float, Device::Cpu));
        let last_value = Tensor::zeros([2], (Kind::Float, Device::Cpu));
        
        let advantages = compute_gae(&rewards, &values, &dones, &last_value, 0.99, 0.95);
        
        assert_eq!(advantages.size(), [4, 2]);
        // With all rewards = 1, values = 0, dones = 0, advantages should be positive
        assert!(advantages.mean(Kind::Float).double_value(&[]) > 0.0);
    }
}
