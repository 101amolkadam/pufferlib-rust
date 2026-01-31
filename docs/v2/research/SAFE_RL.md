# Safe Reinforcement Learning Guide

*Technical specification for constrained policy optimization and safety constraints.*

---

## Overview

Safe RL ensures policies respect safety constraints during both training and deployment. This is critical for real-world applications like robotics, autonomous vehicles, and healthcare.

**Key Concepts:**
- **Constrained MDPs**: Optimize reward subject to constraint satisfaction
- **Shielding**: Runtime safety monitors that override unsafe actions
- **Conservative Updates**: Limit policy change to prevent catastrophic actions

---

## Constrained Policy Optimization

### Lagrangian Relaxation

Convert constrained optimization to unconstrained via Lagrange multipliers:

```
max π [J(π)]  subject to  C(π) ≤ d

becomes:

max π, min λ≥0 [J(π) - λ(C(π) - d)]
```

### File: `crates/pufferlib/src/training/safe.rs`

```rust
//! Safe RL with constraints.

use tch::{nn, Tensor, Kind};

/// Configuration for constrained optimization
#[derive(Clone, Debug)]
pub struct SafeRLConfig {
    /// Cost threshold (constraint limit)
    pub cost_limit: f64,
    /// Initial Lagrange multiplier
    pub initial_lambda: f64,
    /// Learning rate for Lagrange multiplier
    pub lambda_lr: f64,
    /// Maximum Lagrange multiplier (prevent divergence)
    pub lambda_max: f64,
    /// KL constraint for TRPO-style updates
    pub target_kl: f64,
    /// Use penalty vs Lagrangian method
    pub use_lagrangian: bool,
}

impl Default for SafeRLConfig {
    fn default() -> Self {
        Self {
            cost_limit: 25.0,  // Max cumulative cost per episode
            initial_lambda: 0.1,
            lambda_lr: 0.01,
            lambda_max: 100.0,
            target_kl: 0.01,
            use_lagrangian: true,
        }
    }
}

/// Lagrangian-based constrained PPO
pub struct ConstrainedPPO {
    config: SafeRLConfig,
    /// Current Lagrange multiplier
    lambda: f64,
    /// Running cost estimate
    avg_cost: f64,
}

impl ConstrainedPPO {
    pub fn new(config: SafeRLConfig) -> Self {
        Self {
            lambda: config.initial_lambda,
            avg_cost: 0.0,
            config,
        }
    }
    
    /// Compute constrained objective loss
    pub fn compute_loss(
        &self,
        advantages: &Tensor,      // Standard reward advantages
        cost_advantages: &Tensor, // Cost advantages
        log_probs: &Tensor,
        old_log_probs: &Tensor,
        clip_coef: f64,
    ) -> (Tensor, Tensor) {
        let ratio = (log_probs - old_log_probs).exp();
        
        // Standard PPO surrogate
        let surr1 = &ratio * advantages;
        let surr2 = ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef) * advantages;
        let reward_loss = -surr1.min_other(&surr2).mean(Kind::Float);
        
        // Cost surrogate (we want to minimize this)
        let cost_surr = (&ratio * cost_advantages).mean(Kind::Float);
        
        // Combined objective: reward - lambda * cost
        let total_loss = &reward_loss + self.lambda * &cost_surr;
        
        (total_loss, cost_surr)
    }
    
    /// Update Lagrange multiplier based on constraint violation
    pub fn update_lambda(&mut self, episode_cost: f64) {
        // Exponential moving average of cost
        self.avg_cost = 0.9 * self.avg_cost + 0.1 * episode_cost;
        
        // Gradient ascent on lambda
        let violation = self.avg_cost - self.config.cost_limit;
        self.lambda += self.config.lambda_lr * violation;
        
        // Clamp lambda
        self.lambda = self.lambda.max(0.0).min(self.config.lambda_max);
    }
    
    /// Get current constraint satisfaction status
    pub fn is_safe(&self) -> bool {
        self.avg_cost <= self.config.cost_limit
    }
}
```

---

## Cost Value Function

```rust
/// Separate value function for costs
pub struct CostCritic {
    net: nn::Sequential,
}

impl CostCritic {
    pub fn new(vs: &nn::Path, obs_dim: i64, hidden_dim: i64) -> Self {
        let net = nn::seq()
            .add(nn::linear(vs / "cost_fc1", obs_dim, hidden_dim, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "cost_fc2", hidden_dim, hidden_dim, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "cost_fc3", hidden_dim, 1, Default::default()));
        
        Self { net }
    }
    
    pub fn forward(&self, obs: &Tensor) -> Tensor {
        obs.apply(&self.net)
    }
}

/// Compute cost advantages using GAE
pub fn compute_cost_gae(
    costs: &Tensor,
    cost_values: &Tensor,
    dones: &Tensor,
    last_cost_value: &Tensor,
    gamma: f64,
    gae_lambda: f64,
) -> Tensor {
    // Same as reward GAE, but for costs
    use crate::training::compute_gae;
    compute_gae(costs, cost_values, dones, last_cost_value, gamma, gae_lambda)
}
```

---

## Runtime Shielding

```rust
//! Runtime safety monitors.

/// Trait for safety shields
pub trait SafetyShield: Send {
    /// Check if action is safe given current state
    fn is_safe(&self, observation: &Tensor, action: &Tensor) -> bool;
    
    /// Project action to safe region
    fn safe_action(&self, observation: &Tensor, action: &Tensor) -> Tensor;
    
    /// Get safety margin (positive = safe, negative = unsafe)
    fn safety_margin(&self, observation: &Tensor, action: &Tensor) -> f64;
}

/// Simple box constraint shield
pub struct BoxConstraintShield {
    action_low: Tensor,
    action_high: Tensor,
    state_low: Tensor,
    state_high: Tensor,
}

impl BoxConstraintShield {
    pub fn new(
        action_low: &[f64],
        action_high: &[f64],
        state_low: &[f64],
        state_high: &[f64],
    ) -> Self {
        Self {
            action_low: Tensor::from_slice(action_low),
            action_high: Tensor::from_slice(action_high),
            state_low: Tensor::from_slice(state_low),
            state_high: Tensor::from_slice(state_high),
        }
    }
}

impl SafetyShield for BoxConstraintShield {
    fn is_safe(&self, observation: &Tensor, action: &Tensor) -> bool {
        // Check action bounds
        let action_safe = action.ge(&self.action_low).all().int64_value(&[]) == 1
            && action.le(&self.action_high).all().int64_value(&[]) == 1;
        
        // Check state bounds
        let state_safe = observation.ge(&self.state_low).all().int64_value(&[]) == 1
            && observation.le(&self.state_high).all().int64_value(&[]) == 1;
        
        action_safe && state_safe
    }
    
    fn safe_action(&self, _observation: &Tensor, action: &Tensor) -> Tensor {
        action.clamp(&self.action_low, &self.action_high)
    }
    
    fn safety_margin(&self, observation: &Tensor, action: &Tensor) -> f64 {
        // Minimum distance to constraint boundary
        let action_margin = action.sub(&self.action_low).min().double_value(&[])
            .min(self.action_high.sub(action).min().double_value(&[]));
        
        let state_margin = observation.sub(&self.state_low).min().double_value(&[])
            .min(self.state_high.sub(observation).min().double_value(&[]));
        
        action_margin.min(state_margin)
    }
}

/// Neural network based safety classifier
pub struct LearnedShield {
    classifier: nn::Sequential,
    threshold: f64,
}

impl LearnedShield {
    pub fn new(vs: &nn::Path, input_dim: i64, hidden_dim: i64, threshold: f64) -> Self {
        let classifier = nn::seq()
            .add(nn::linear(vs / "shield_fc1", input_dim, hidden_dim, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs / "shield_fc2", hidden_dim, 1, Default::default()))
            .add_fn(|x| x.sigmoid());
        
        Self { classifier, threshold }
    }
    
    /// Train shield from (state, action, safe?) tuples
    pub fn train(
        &mut self,
        states: &Tensor,
        actions: &Tensor,
        labels: &Tensor,  // 1.0 = safe, 0.0 = unsafe
        optimizer: &mut nn::Optimizer,
    ) -> f64 {
        let input = Tensor::cat(&[states, actions], 1);
        let pred = input.apply(&self.classifier);
        
        let loss = pred.binary_cross_entropy_with_logits::<Tensor>(
            labels, None, None, tch::Reduction::Mean,
        );
        
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        
        loss.double_value(&[])
    }
}

impl SafetyShield for LearnedShield {
    fn is_safe(&self, observation: &Tensor, action: &Tensor) -> bool {
        let input = Tensor::cat(&[observation, action], 0).unsqueeze(0);
        let pred = tch::no_grad(|| input.apply(&self.classifier));
        pred.double_value(&[0]) > self.threshold
    }
    
    fn safe_action(&self, observation: &Tensor, action: &Tensor) -> Tensor {
        if self.is_safe(observation, action) {
            action.shallow_clone()
        } else {
            // Fallback to safe default (e.g., zero action or previous action)
            Tensor::zeros_like(action)
        }
    }
    
    fn safety_margin(&self, observation: &Tensor, action: &Tensor) -> f64 {
        let input = Tensor::cat(&[observation, action], 0).unsqueeze(0);
        let pred = tch::no_grad(|| input.apply(&self.classifier));
        pred.double_value(&[0]) - self.threshold
    }
}
```

---

## Integration with Trainer

```rust
/// Safe Trainer with constraints and shielding
pub struct SafeTrainer<P: Policy + HasVarStore, V: VecEnvBackend, S: SafetyShield> {
    trainer: Trainer<P, V>,
    cost_critic: CostCritic,
    constrained_ppo: ConstrainedPPO,
    shield: Option<S>,
    config: SafeRLConfig,
}

impl<P: Policy + HasVarStore, V: VecEnvBackend, S: SafetyShield> SafeTrainer<P, V, S> {
    /// Collect rollout with shielded actions
    pub fn collect_safe_rollout(&mut self) {
        // Similar to regular rollout, but apply shield
        let action = if let Some(shield) = &self.shield {
            shield.safe_action(&obs, &raw_action)
        } else {
            raw_action
        };
        
        // Track costs
        let cost = self.compute_step_cost(&obs, &action);
        self.cost_buffer.push(cost);
    }
    
    /// Update with constrained objective
    pub fn update(&mut self) -> SafeTrainMetrics {
        // Compute cost advantages
        let cost_advantages = compute_cost_gae(/*...*/);
        
        // Constrained PPO update
        let (loss, cost_loss) = self.constrained_ppo.compute_loss(
            &advantages,
            &cost_advantages,
            &log_probs,
            &old_log_probs,
            self.config.clip_coef,
        );
        
        // Update lambda based on episode costs
        self.constrained_ppo.update_lambda(self.avg_episode_cost);
        
        SafeTrainMetrics {
            reward_loss: loss.double_value(&[]),
            cost_loss: cost_loss.double_value(&[]),
            lambda: self.constrained_ppo.lambda,
            avg_cost: self.constrained_ppo.avg_cost,
            is_safe: self.constrained_ppo.is_safe(),
        }
    }
}
```

---

## Example: Safe CartPole

```rust
fn safe_cartpole_training() {
    // Define safety constraint: pole angle < 0.2 radians
    let shield = BoxConstraintShield::new(
        &[-1.0, -1.0],  // action bounds
        &[1.0, 1.0],
        &[-2.4, -10.0, -0.2, -10.0],  // state bounds (position, velocity, angle, angular_vel)
        &[2.4, 10.0, 0.2, 10.0],
    );
    
    let safe_config = SafeRLConfig {
        cost_limit: 10.0,  // Max 10 safety violations per episode
        ..Default::default()
    };
    
    let mut trainer = SafeTrainer::new(
        policy, env, trainer_config, cost_critic, safe_config, Some(shield),
    );
    
    for epoch in 0..1000 {
        let metrics = trainer.update();
        
        if metrics.is_safe {
            println!("Epoch {}: Safe! λ={:.3}, cost={:.2}", 
                epoch, metrics.lambda, metrics.avg_cost);
        } else {
            println!("Epoch {}: UNSAFE! Increasing penalty λ={:.3}", 
                epoch, metrics.lambda);
        }
    }
}
```

---

## Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lambda_update() {
        let mut cppo = ConstrainedPPO::new(SafeRLConfig {
            cost_limit: 10.0,
            initial_lambda: 0.1,
            lambda_lr: 0.1,
            ..Default::default()
        });
        
        // Violate constraint
        cppo.update_lambda(15.0);
        assert!(cppo.lambda > 0.1);
        
        // Satisfy constraint
        cppo.update_lambda(5.0);
        // Lambda should decrease (but stay positive)
    }
    
    #[test]
    fn test_box_shield() {
        let shield = BoxConstraintShield::new(
            &[-1.0], &[1.0], &[-1.0], &[1.0],
        );
        
        let obs = Tensor::from_slice(&[0.0f32]);
        let safe_action = Tensor::from_slice(&[0.5f32]);
        let unsafe_action = Tensor::from_slice(&[1.5f32]);
        
        assert!(shield.is_safe(&obs, &safe_action));
        assert!(!shield.is_safe(&obs, &unsafe_action));
        
        let projected = shield.safe_action(&obs, &unsafe_action);
        assert!(shield.is_safe(&obs, &projected));
    }
}
```

---

## References

- [Constrained Policy Optimization (CPO)](https://arxiv.org/abs/1705.10528)
- [TRPO-Lagrangian](https://arxiv.org/abs/1801.08757)
- [Safe RL Survey](https://arxiv.org/abs/2205.00842)
- [Shielding in RL](https://arxiv.org/abs/1709.02753)

---

*Last updated: 2026-01-28*
