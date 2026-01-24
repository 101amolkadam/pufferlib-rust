//! Tuple observation/action space

use super::{DynSpace, Space};
use rand::Rng;

/// Tuple space containing an ordered list of sub-spaces
#[derive(Clone, Debug)]
pub struct Tuple {
    /// Ordered sub-spaces
    pub spaces: Vec<DynSpace>,
    /// Cached total shape
    shape: Vec<usize>,
}

impl Tuple {
    /// Create a new tuple space
    pub fn new(spaces: Vec<DynSpace>) -> Self {
        let total: usize = spaces
            .iter()
            .map(|s| s.shape().iter().product::<usize>())
            .sum();
        Self {
            spaces,
            shape: vec![total],
        }
    }
}

impl Space for Tuple {
    type Sample = Vec<ndarray::ArrayD<f32>>;

    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Sample {
        self.spaces.iter().map(|s| s.sample(rng)).collect()
    }

    fn contains(&self, value: &Self::Sample) -> bool {
        if value.len() != self.spaces.len() {
            return false;
        }
        for (v, s) in value.iter().zip(self.spaces.iter()) {
            if !s.contains(v) {
                return false;
            }
        }
        true
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::{Discrete, Box as BoxSpace};

    #[test]
    fn test_tuple_creation() {
        let tuple = Tuple::new(vec![
            DynSpace::Discrete(Discrete::new(2)),
            DynSpace::Box(BoxSpace::uniform(&[2], 0.0, 1.0)),
        ]);
        assert_eq!(tuple.shape(), &[3]); // 1 (discrete) + 2 (box)
    }
}
