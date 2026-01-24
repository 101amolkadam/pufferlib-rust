//! Dict (dictionary) observation/action space

use std::collections::HashMap;
use super::{DynSpace, Space};
use rand::Rng;

/// Dictionary space containing named sub-spaces
#[derive(Clone, Debug)]
pub struct Dict {
    /// Named sub-spaces
    pub spaces: HashMap<String, DynSpace>,
    /// Cached total shape (sum of all sub-space shapes)
    shape: Vec<usize>,
}

impl Dict {
    /// Create a new dict space
    pub fn new(spaces: HashMap<String, DynSpace>) -> Self {
        // Calculate total flattened size
        let total: usize = spaces.values().map(|s| s.shape().iter().product::<usize>()).sum();
        Self {
            spaces,
            shape: vec![total],
        }
    }
    
    /// Create from a list of (name, space) pairs
    pub fn from_pairs(pairs: Vec<(&str, DynSpace)>) -> Self {
        let spaces: HashMap<_, _> = pairs.into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        Self::new(spaces)
    }
    
    /// Get a sub-space by name
    pub fn get(&self, name: &str) -> Option<&DynSpace> {
        self.spaces.get(name)
    }
    
    /// Get all space names
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.spaces.keys()
    }
}

impl Space for Dict {
    type Sample = HashMap<String, ndarray::ArrayD<f32>>;
    
    fn sample<R: Rng>(&self, rng: &mut R) -> Self::Sample {
        self.spaces.iter()
            .map(|(k, v)| (k.clone(), v.sample(rng)))
            .collect()
    }
    
    fn contains(&self, value: &Self::Sample) -> bool {
        if value.len() != self.spaces.len() {
            return false;
        }
        for (k, v) in &self.spaces {
            match value.get(k) {
                Some(sample) => {
                    if !v.contains(sample) {
                        return false;
                    }
                }
                None => return false,
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
    fn test_dict_creation() {
        let dict = Dict::from_pairs(vec![
            ("position", DynSpace::Box(BoxSpace::uniform(&[3], -1.0, 1.0))),
            ("action", DynSpace::Discrete(Discrete::new(4))),
        ]);
        
        assert!(dict.get("position").is_some());
        assert!(dict.get("action").is_some());
        assert!(dict.get("unknown").is_none());
    }

    #[test]
    fn test_dict_contains() {
        let dict = Dict::from_pairs(vec![
            ("pos", DynSpace::Discrete(Discrete::new(2))),
        ]);

        let mut valid = HashMap::new();
        valid.insert("pos".to_string(), ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1]), 0.0));
        assert!(dict.contains(&valid));

        let mut invalid_val = HashMap::new();
        invalid_val.insert("pos".to_string(), ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1]), 2.0));
        assert!(!dict.contains(&invalid_val));

        let mut invalid_key = HashMap::new();
        invalid_key.insert("wrong".to_string(), ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1]), 0.0));
        assert!(!dict.contains(&invalid_key));
    }
}
