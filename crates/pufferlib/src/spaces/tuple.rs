//! Tuple observation/action space.

use super::{DynSpace, Space};
use crate::types::{vec, Vec};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Tuple space containing an ordered list of subspaces
#[derive(Clone, Debug, Serialize, Deserialize)]
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

    fn flatten_to(&self, value: &Self::Sample, out: &mut [f32]) {
        let mut offset = 0;
        for (space, sample) in self.spaces.iter().zip(value.iter()) {
            let size = space.shape().iter().product::<usize>();
            space.flatten_to(sample, &mut out[offset..offset + size]);
            offset += size;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::{Box as BoxSpace, Discrete};

    #[test]
    fn test_tuple_creation() {
        let tuple = Tuple::new(vec![
            DynSpace::Discrete(Discrete::new(2)),
            DynSpace::Box(BoxSpace::uniform(&[2], 0.0, 1.0)),
        ]);
        assert_eq!(tuple.shape(), &[3]); // 1 (discrete) + 2 (box)
    }
}
