//! MultiDiscrete action/observation space

use rand::Rng;
use super::Space;

/// MultiDiscrete space for multiple discrete action dimensions
/// 
/// Each dimension i has nvec[i] possible values: {0, 1, ..., nvec[i]-1}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiDiscrete {
    /// Number of values for each dimension
    pub nvec: Vec<usize>,
    /// Cached shape (just the length of nvec)
    shape: Vec<usize>,
}

impl MultiDiscrete {
    /// Create a new multi-discrete space
    pub fn new(nvec: Vec<usize>) -> Self {
        assert!(!nvec.is_empty(), "MultiDiscrete must have at least 1 dimension");
        assert!(nvec.iter().all(|&n| n > 0), "All dimensions must have at least 1 element");
        let shape = vec![nvec.len()];
        Self { nvec, shape }
    }
    
    /// Create from a slice
    pub fn from_slice(nvec: &[usize]) -> Self {
        Self::new(nvec.to_vec())
    }
    
    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.nvec.len()
    }
}

impl Space for MultiDiscrete {
    type Sample = Vec<usize>;
    
    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Sample {
        self.nvec.iter().map(|&n| rng.gen_range(0..n)).collect()
    }
    
    fn contains(&self, value: &Self::Sample) -> bool {
        if value.len() != self.nvec.len() {
            return false;
        }
        value.iter().zip(self.nvec.iter()).all(|(&v, &n)| v < n)
    }
    
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    fn num_elements(&self) -> usize {
        self.nvec.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    
    #[test]
    fn test_multi_discrete_sample() {
        let space = MultiDiscrete::new(vec![3, 4, 5]);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        
        for _ in 0..100 {
            let sample = space.sample(&mut rng);
            assert!(space.contains(&sample));
            assert_eq!(sample.len(), 3);
        }
    }
    
    #[test]
    fn test_multi_discrete_contains() {
        let space = MultiDiscrete::new(vec![3, 4]);
        assert!(space.contains(&vec![0, 0]));
        assert!(space.contains(&vec![2, 3]));
        assert!(!space.contains(&vec![3, 0])); // First dim out of range
        assert!(!space.contains(&vec![0, 4])); // Second dim out of range
        assert!(!space.contains(&vec![0])); // Wrong length
    }

    #[test]
    fn test_multi_discrete_properties() {
        let space = MultiDiscrete::new(vec![2, 2, 2]);
        assert_eq!(space.ndim(), 3);
        assert_eq!(space.num_elements(), 3);
        assert_eq!(space.shape(), &[3]);
    }
}
