//! Verified policies integration for formal methods.

use crate::types::String;

/// Trait for policies that support formal verification
pub trait VerifiablePolicy {
    /// Export policy to VNN-LIB format for formal verification
    fn export_vnnlib(&self, path: &str) -> Result<(), Box<dyn std::error::Error>>;

    /// Get safety properties for this policy
    fn safety_properties(&self) -> Vec<String>;
}

/// A policy that has been verified against a set of properties
pub struct VerifiedPolicy<P> {
    pub policy: P,
    pub verified_properties: Vec<String>,
}

impl<P> VerifiedPolicy<P> {
    pub fn new(policy: P, properties: Vec<String>) -> Self {
        Self {
            policy,
            verified_properties: properties,
        }
    }
}
