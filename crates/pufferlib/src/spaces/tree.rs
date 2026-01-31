//! Space metadata for recursive flattening/unflattening.

use super::{DynSpace, Space};
use crate::types::{Box, HashMap, String, Vec};

/// Tree structure representing the nested layout of a complex space
#[derive(Clone, Debug)]
pub enum SpaceTree {
    /// Leaf node representing a primitive space (Box, Discrete, etc.)
    Leaf {
        /// Type of the primitive space
        space: Box<DynSpace>,
        /// Flattened offset in the destination buffer
        offset: usize,
        /// Size of the flattened representation
        size: usize,
    },
    /// Branch node representing an ordered collection (Tuple)
    Tuple {
        /// Ordered sub-trees
        children: Vec<SpaceTree>,
        /// Total flattened size of this branch
        size: usize,
    },
    /// Branch node representing a named collection (Dict)
    Dict {
        /// Named sub-trees (keys in alphabetical order for deterministic flattening)
        children: Vec<(String, SpaceTree)>,
        /// Total flattened size of this branch
        size: usize,
    },
}

impl SpaceTree {
    /// Create a SpaceTree from a DynSpace
    pub fn from_space(space: &DynSpace) -> Self {
        Self::build(space, 0)
    }

    fn build(space: &DynSpace, offset: usize) -> Self {
        match space {
            DynSpace::Box(s) => {
                let size = s.num_elements();
                SpaceTree::Leaf {
                    space: Box::new(space.clone()),
                    offset,
                    size,
                }
            }
            DynSpace::Discrete(_) | DynSpace::MultiDiscrete(_) => {
                let size = space.shape().iter().product();
                SpaceTree::Leaf {
                    space: Box::new(space.clone()),
                    offset,
                    size,
                }
            }
            DynSpace::Dict(s) => {
                let mut children = Vec::new();
                let mut current_offset = offset;

                // Sort keys for deterministic flattening
                let mut keys: Vec<_> = s.spaces.keys().cloned().collect();
                keys.sort();

                for k in keys {
                    let child_space = s.spaces.get(&k).unwrap();
                    let child_tree = SpaceTree::build(child_space, current_offset);
                    current_offset += child_tree.size();
                    children.push((k, child_tree));
                }

                SpaceTree::Dict {
                    children,
                    size: current_offset - offset,
                }
            }
            DynSpace::Tuple(s) => {
                let mut children = Vec::new();
                let mut current_offset = offset;

                for child_space in &s.spaces {
                    let child_tree = SpaceTree::build(child_space, current_offset);
                    current_offset += child_tree.size();
                    children.push(child_tree);
                }

                SpaceTree::Tuple {
                    children,
                    size: current_offset - offset,
                }
            }
        }
    }

    /// Get the total flattened size of this tree
    pub fn size(&self) -> usize {
        match self {
            SpaceTree::Leaf { size, .. } => *size,
            SpaceTree::Tuple { size, .. } => *size,
            SpaceTree::Dict { size, .. } => *size,
        }
    }

    /// Flatten a nested observation into a pre-allocated buffer
    pub fn flatten(&self, obs: &crate::env::Observation, buf: &mut [f32]) {
        match (self, obs) {
            (SpaceTree::Leaf { offset, size, .. }, crate::env::Observation::Array(arr)) => {
                let data = arr.as_slice().unwrap();
                buf[*offset..*offset + *size].copy_from_slice(data);
            }
            (SpaceTree::Dict { children, .. }, crate::env::Observation::Dict(map)) => {
                for (name, child_tree) in children {
                    if let Some(child_obs) = map.get(name) {
                        child_tree.flatten(child_obs, buf);
                    }
                }
            }
            (SpaceTree::Tuple { children, .. }, crate::env::Observation::Tuple(list)) => {
                for (child_tree, child_obs) in children.iter().zip(list.iter()) {
                    child_tree.flatten(child_obs, buf);
                }
            }
            _ => panic!("SpaceTree/Observation mismatch during flattening"),
        }
    }

    /// Unflatten a slice into a structured Action
    pub fn unflatten(&self, buf: &[f32]) -> crate::env::Action {
        match self {
            SpaceTree::Leaf { offset, size, .. } => {
                let vec = buf[*offset..*offset + *size].to_vec();
                crate::env::Action::Array(
                    ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[*size]), vec).unwrap(),
                )
            }
            SpaceTree::Dict { children, .. } => {
                let mut map = HashMap::new();
                for (name, child_tree) in children {
                    map.insert(name.clone(), child_tree.unflatten(buf));
                }
                crate::env::Action::Dict(map)
            }
            SpaceTree::Tuple { children, .. } => {
                let list = children.iter().map(|c| c.unflatten(buf)).collect();
                crate::env::Action::Tuple(list)
            }
        }
    }
}
