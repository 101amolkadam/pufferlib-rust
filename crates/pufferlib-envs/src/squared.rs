//! Squared grid navigation environment.

use ndarray::{Array2, ArrayD, IxDyn};
use pufferlib::env::{PufferEnv, EnvInfo, StepResult};
use pufferlib::spaces::{DynSpace, Discrete, Box as BoxSpace};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

/// Squared grid navigation environment
///
/// Agent starts at center of a grid and must reach targets on the perimeter.
/// Observation is the grid state, action is direction to move.
pub struct Squared {
    /// Distance from center to perimeter
    distance_to_target: usize,
    /// Number of targets
    num_targets: usize,
    /// Grid size (2 * distance_to_target + 1)
    grid_size: usize,
    /// Maximum steps per episode
    max_ticks: u32,
    /// Current grid state
    grid: Array2<f32>,
    /// Agent position (x, y)
    agent_pos: (usize, usize),
    /// Remaining targets
    targets: Vec<(usize, usize)>,
    /// Current tick
    tick: u32,
    /// RNG
    rng: StdRng,
}

// 8 movement directions: N, S, W, E, NE, NW, SE, SW
const MOVES: [(i32, i32); 8] = [
    (0, -1), (0, 1), (-1, 0), (1, 0),
    (1, -1), (-1, -1), (1, 1), (-1, 1),
];

impl Squared {
    /// Create a new squared environment
    pub fn new(distance_to_target: usize) -> Self {
        Self::with_targets(distance_to_target, 4 * distance_to_target)
    }
    
    /// Create with specific number of targets
    pub fn with_targets(distance_to_target: usize, num_targets: usize) -> Self {
        let grid_size = 2 * distance_to_target + 1;
        
        Self {
            distance_to_target,
            num_targets,
            grid_size,
            max_ticks: (num_targets * distance_to_target) as u32,
            grid: Array2::zeros((grid_size, grid_size)),
            agent_pos: (distance_to_target, distance_to_target),
            targets: Vec::new(),
            tick: 0,
            rng: StdRng::from_entropy(),
        }
    }
    
    fn possible_targets(&self) -> Vec<(usize, usize)> {
        let mut targets = Vec::new();
        for x in 0..self.grid_size {
            for y in 0..self.grid_size {
                if x == 0 || y == 0 || x == self.grid_size - 1 || y == self.grid_size - 1 {
                    targets.push((x, y));
                }
            }
        }
        targets
    }
}

impl PufferEnv for Squared {
    fn observation_space(&self) -> DynSpace {
        let size = self.grid_size * self.grid_size;
        DynSpace::Box(BoxSpace::uniform(&[size], -1.0, 1.0))
    }
    
    fn action_space(&self) -> DynSpace {
        DynSpace::Discrete(Discrete::new(8))
    }
    
    fn reset(&mut self, seed: Option<u64>) -> (ArrayD<f32>, EnvInfo) {
        if let Some(s) = seed {
            self.rng = StdRng::seed_from_u64(s);
        }
        
        // Reset grid and agent
        self.grid.fill(0.0);
        self.agent_pos = (self.distance_to_target, self.distance_to_target);
        self.grid[self.agent_pos] = -1.0;
        self.tick = 0;
        
        // Place random targets on perimeter
        let mut possible = self.possible_targets();
        possible.shuffle(&mut self.rng);
        self.targets = possible.into_iter().take(self.num_targets).collect();
        
        for &(x, y) in &self.targets {
            self.grid[(x, y)] = 1.0;
        }
        
        let obs = ArrayD::from_shape_vec(
            IxDyn(&[self.grid_size * self.grid_size]),
            self.grid.iter().copied().collect()
        ).unwrap();
        
        (obs, EnvInfo::new())
    }
    
    fn step(&mut self, action: &ArrayD<f32>) -> StepResult {
        let action_idx = action.iter().next().unwrap().round() as usize;
        assert!(action_idx < 8);
        
        let (ax, ay) = self.agent_pos;
        self.grid[(ax, ay)] = 0.0;
        
        let (dx, dy) = MOVES[action_idx];
        let nx = (ax as i32 + dx).max(0).min(self.grid_size as i32 - 1) as usize;
        let ny = (ay as i32 + dy).max(0).min(self.grid_size as i32 - 1) as usize;
        
        // Compute reward based on distance to nearest target
        let min_dist = self.targets.iter()
            .map(|&(tx, ty)| {
                let dx = (nx as i32 - tx as i32).abs();
                let dy = (ny as i32 - ty as i32).abs();
                dx.max(dy)
            })
            .min()
            .unwrap_or(0);
        
        let reward = 1.0 - min_dist as f32 / self.distance_to_target as f32;
        
        // Check if hit target
        if let Some(idx) = self.targets.iter().position(|&t| t == (nx, ny)) {
            self.targets.remove(idx);
        }
        
        // Reset position if at perimeter
        let dist_from_origin = (nx as i32 - self.distance_to_target as i32).abs()
            .max((ny as i32 - self.distance_to_target as i32).abs());
        
        if dist_from_origin >= self.distance_to_target as i32 {
            self.agent_pos = (self.distance_to_target, self.distance_to_target);
        } else {
            self.agent_pos = (nx, ny);
        }
        
        self.grid[self.agent_pos] = -1.0;
        self.tick += 1;
        
        let done = self.tick >= self.max_ticks;
        let score = (self.num_targets - self.targets.len()) as f32 / self.num_targets as f32;
        
        let obs = ArrayD::from_shape_vec(
            IxDyn(&[self.grid_size * self.grid_size]),
            self.grid.iter().copied().collect()
        ).unwrap();
        
        let info = if done {
            EnvInfo::new().with_extra("score", score)
        } else {
            EnvInfo::new()
        };
        
        StepResult {
            observation: obs,
            reward,
            terminated: done,
            truncated: false,
            info,
        }
    }
    
    fn render(&self) -> Option<String> {
        let mut lines = Vec::new();
        for y in 0..self.grid_size {
            let mut line = String::new();
            for x in 0..self.grid_size {
                let val = self.grid[(x, y)];
                if val < 0.0 {
                    line.push_str("🟢");  // Agent
                } else if val > 0.0 {
                    line.push_str("🎯");  // Target
                } else {
                    line.push_str("⬛");  // Empty
                }
            }
            lines.push(line);
        }
        Some(lines.join("\n"))
    }
    
    fn is_done(&self) -> bool {
        self.tick >= self.max_ticks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_squared_creation() {
        let env = Squared::new(2);
        assert_eq!(env.grid_size, 5);  // 2*2+1 = 5
    }
    
    #[test]
    fn test_squared_reset() {
        let mut env = Squared::new(2);
        let (obs, _) = env.reset(Some(42));
        
        assert_eq!(obs.len(), 25);  // 5x5 grid
    }
    
    #[test]
    fn test_squared_step() {
        let mut env = Squared::new(2);
        env.reset(Some(42));
        
        let action = ArrayD::from_elem(IxDyn(&[1]), 0.0);  // Move north
        let result = env.step(&action);
        
        assert_eq!(result.observation.len(), 25);
    }
}
