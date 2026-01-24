//! Box (continuous) observation/action space

use ndarray::{ArrayD, IxDyn};
use rand::Rng;
use rand_distr::{Distribution, Uniform};
use super::Space;

/// Box space for continuous values with bounds
#[derive(Clone, Debug)]
pub struct Box {
    /// Lower bound for each element
    pub low: ArrayD<f32>,
    /// Upper bound for each element
    pub high: ArrayD<f32>,
    /// Shape of the space
    shape: Vec<usize>,
}

impl Box {
    /// Create a new box space with given bounds
    pub fn new(low: ArrayD<f32>, high: ArrayD<f32>) -> Self {
        assert_eq!(low.shape(), high.shape(), "Low and high must have same shape");
        let shape = low.shape().to_vec();
        Self { low, high, shape }
    }
    
    /// Create a box space with uniform bounds
    pub fn uniform(shape: &[usize], low: f32, high: f32) -> Self {
        let low_arr = ArrayD::from_elem(IxDyn(shape), low);
        let high_arr = ArrayD::from_elem(IxDyn(shape), high);
        Self::new(low_arr, high_arr)
    }
    
    /// Create a box space from -inf to +inf (unbounded)
    pub fn unbounded(shape: &[usize]) -> Self {
        Self::uniform(shape, f32::NEG_INFINITY, f32::INFINITY)
    }
    
    /// Create a unit box [0, 1] for all elements
    pub fn unit(shape: &[usize]) -> Self {
        Self::uniform(shape, 0.0, 1.0)
    }
    
    /// Create a symmetric box [-1, 1] for all elements
    pub fn symmetric(shape: &[usize]) -> Self {
        Self::uniform(shape, -1.0, 1.0)
    }
}

impl Space for Box {
    type Sample = ArrayD<f32>;
    
    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Sample {
        let mut result = ArrayD::zeros(IxDyn(&self.shape));
        for (_i, ((&l, &h), r)) in self.low.iter()
            .zip(self.high.iter())
            .zip(result.iter_mut())
            .enumerate()
        {
            let dist = Uniform::new(l, h);
            *r = dist.sample(rng);
        }
        result
    }
    
    fn contains(&self, value: &Self::Sample) -> bool {
        if value.shape() != self.low.shape() {
            return false;
        }
        value.iter()
            .zip(self.low.iter())
            .zip(self.high.iter())
            .all(|((&v, &l), &h)| v >= l && v <= h)
    }
    
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    
    #[test]
    fn test_box_sample() {
        let space = Box::uniform(&[3, 4], -1.0, 1.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        
        for _ in 0..100 {
            let sample = space.sample(&mut rng);
            assert!(space.contains(&sample));
            assert_eq!(sample.shape(), &[3, 4]);
        }
    }
    
    #[test]
    fn test_box_contains() {
        let space = Box::uniform(&[2], 0.0, 1.0);
        let valid = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 0.5]).unwrap();
        let invalid = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.5, 0.5]).unwrap();
        
        assert!(space.contains(&valid));
        assert!(!space.contains(&invalid));
    }
}
