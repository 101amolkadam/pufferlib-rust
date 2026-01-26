//! Observation and action space types.
//!
//! Provides Gymnasium-compatible space definitions for reinforcement learning.

mod r#box;
mod dict;
mod discrete;
mod multi_discrete;
mod tree;
mod tuple;

pub use dict::Dict;
pub use discrete::Discrete;
pub use multi_discrete::MultiDiscrete;
pub use r#box::Box;
pub use tree::SpaceTree;
pub use tuple::Tuple;

use ndarray::ArrayD;
use rand::Rng;

/// Trait for observation and action spaces
pub trait Space: Clone + Send + Sync {
    /// The type of samples from this space
    type Sample;

    /// Sample a random element from this space
    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Sample;

    /// Check if a value is contained in this space
    fn contains(&self, value: &Self::Sample) -> bool;

    /// Get the shape of samples from this space
    fn shape(&self) -> &[usize];

    /// Get the total number of elements in a sample
    fn num_elements(&self) -> usize {
        self.shape().iter().product()
    }

    /// Flatten a sample into a provided buffer
    fn flatten_to(&self, value: &Self::Sample, out: &mut [f32]);
}

/// Enum for dynamic space types
#[derive(Clone, Debug)]
pub enum DynSpace {
    Discrete(Discrete),
    MultiDiscrete(MultiDiscrete),
    Box(Box),
    Dict(Dict),
    Tuple(Tuple),
}

impl DynSpace {
    /// Get the shape of this space
    pub fn shape(&self) -> Vec<usize> {
        match self {
            DynSpace::Discrete(s) => s.shape().to_vec(),
            DynSpace::MultiDiscrete(s) => s.shape().to_vec(),
            DynSpace::Box(s) => s.shape().to_vec(),
            DynSpace::Dict(s) => s.shape().to_vec(),
            DynSpace::Tuple(s) => s.shape().to_vec(),
        }
    }

    pub fn flatten_to(&self, value: &ndarray::ArrayD<f32>, out: &mut [f32]) {
        match self {
            DynSpace::Discrete(s) => {
                let v = value.iter().next().unwrap().round() as usize;
                s.flatten_to(&v, out);
            }
            DynSpace::MultiDiscrete(s) => {
                let v: Vec<usize> = value.iter().map(|&x| x.round() as usize).collect();
                s.flatten_to(&v, out);
            }
            DynSpace::Box(s) => s.flatten_to(value, out),
            DynSpace::Dict(_) => {
                panic!("Dict space should be flattened recursively via nested samples")
            }
            DynSpace::Tuple(_) => {
                panic!("Tuple space should be flattened recursively via nested samples")
            }
        }
    }

    /// Sample from this space
    pub fn sample<R: Rng>(&self, rng: &mut R) -> ArrayD<f32> {
        match self {
            DynSpace::Discrete(s) => {
                let v = s.sample(rng);
                ArrayD::from_elem(ndarray::IxDyn(&[1]), v as f32)
            }
            DynSpace::MultiDiscrete(s) => {
                let v = s.sample(rng);
                ArrayD::from_shape_vec(
                    ndarray::IxDyn(&[v.len()]),
                    v.into_iter().map(|x| x as f32).collect(),
                )
                .unwrap()
            }
            DynSpace::Box(s) => s.sample(rng),
            DynSpace::Dict(s) => {
                let sample = s.sample(rng);
                let mut flattened = Vec::new();
                for (_, v) in sample {
                    flattened.extend(v.into_iter());
                }
                ArrayD::from_shape_vec(ndarray::IxDyn(&[flattened.len()]), flattened).unwrap()
            }
            DynSpace::Tuple(s) => {
                let sample = s.sample(rng);
                let mut flattened = Vec::new();
                for v in sample {
                    flattened.extend(v.into_iter());
                }
                ArrayD::from_shape_vec(ndarray::IxDyn(&[flattened.len()]), flattened).unwrap()
            }
        }
    }

    /// Check if this space contains the value
    pub fn contains(&self, value: &ArrayD<f32>) -> bool {
        match self {
            DynSpace::Discrete(s) => {
                if value.len() != 1 {
                    return false;
                }
                let v = value.iter().next().unwrap().round() as usize;
                s.contains(&v)
            }
            DynSpace::MultiDiscrete(s) => {
                if value.len() != s.nvec.len() {
                    return false;
                }
                let v: Vec<usize> = value.iter().map(|&x| x.round() as usize).collect();
                s.contains(&v)
            }
            DynSpace::Box(s) => s.contains(value),
            DynSpace::Dict(_) => false, // TODO: Support recursive contains
            DynSpace::Tuple(_) => false,
        }
    }
}
