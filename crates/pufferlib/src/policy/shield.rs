//! Safety shielding for RL policies.

#[cfg(feature = "torch")]
use crate::policy::{Distribution, HasVarStore, Policy};
#[cfg(feature = "torch")]
use tch::Tensor;

/// Trait for safety shields that can monitor and correct actions
#[cfg(feature = "torch")]
pub trait SafetyShield: Send {
    /// Correct an action if it violates safety constraints
    fn shield_action(&self, observations: &Tensor, action: &Tensor) -> Tensor;
}

/// Wrapper that applies a safety shield to any policy
#[cfg(feature = "torch")]
pub struct ShieldedPolicy<P: Policy, S: SafetyShield> {
    pub policy: P,
    pub shield: S,
}

#[cfg(feature = "torch")]
impl<P: Policy, S: SafetyShield> ShieldedPolicy<P, S> {
    pub fn new(policy: P, shield: S) -> Self {
        Self { policy, shield }
    }
}

#[cfg(feature = "torch")]
impl<P: Policy + HasVarStore, S: SafetyShield> HasVarStore for ShieldedPolicy<P, S> {
    fn var_store(&self) -> &tch::nn::VarStore {
        self.policy.var_store()
    }
    fn var_store_mut(&mut self) -> &mut tch::nn::VarStore {
        self.policy.var_store_mut()
    }
}

#[cfg(feature = "torch")]
impl<P: Policy, S: SafetyShield> Policy for ShieldedPolicy<P, S> {
    fn forward(
        &self,
        observations: &Tensor,
        state: &Option<Vec<Tensor>>,
    ) -> (Distribution, Tensor, Option<Vec<Tensor>>) {
        let (dist, value, next_state) = self.policy.forward(observations, state);

        // Note: Shielding usually happens during inference/sampling
        // or by modifying the distribution.
        (dist, value, next_state)
    }

    fn find_distribution(
        &self,
        observations: &Tensor,
        state: &Option<Vec<Tensor>>,
    ) -> (Distribution, Tensor, Option<Vec<Tensor>>) {
        self.policy.forward(observations, state)
    }

    fn initial_state(&self, batch_size: i64) -> Option<Vec<Tensor>> {
        self.policy.initial_state(batch_size)
    }
}
