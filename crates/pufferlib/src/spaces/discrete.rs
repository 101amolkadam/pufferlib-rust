//! Discrete action/observation space

use super::Space;
use rand::Rng;

/// Discrete space with n possible values: {0, 1, ..., n-1}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Discrete {
    /// Number of possible values
    pub n: usize,
    /// Cached shape
    shape: Vec<usize>,
}

impl Discrete {
    /// Create a new discrete space with n values
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "Discrete space must have at least 1 element");
        Self { n, shape: vec![1] }
    }
}

impl Space for Discrete {
    type Sample = usize;

    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Sample {
        rng.gen_range(0..self.n)
    }

    fn contains(&self, value: &Self::Sample) -> bool {
        *value < self.n
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn num_elements(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_discrete_sample() {
        let space = Discrete::new(4);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let sample = space.sample(&mut rng);
            assert!(space.contains(&sample));
            assert!(sample < 4);
        }
    }

    #[test]
    fn test_discrete_contains() {
        let space = Discrete::new(5);
        assert!(space.contains(&0));
        assert!(space.contains(&4));
        assert!(!space.contains(&5));
    }

    #[test]
    fn test_discrete_single_element() {
        let space = Discrete::new(1);
        assert!(space.contains(&0));
        assert!(!space.contains(&1));
        assert_eq!(space.n, 1);
        assert_eq!(space.shape(), &[1]);
    }
}
