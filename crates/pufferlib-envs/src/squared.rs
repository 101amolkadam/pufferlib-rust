//! Squared grid navigation environment.

use ndarray::{Array2, ArrayD, IxDyn};
use pufferlib::env::{EnvInfo, PufferEnv, StepResult};
use pufferlib::spaces::{Box as BoxSpace, Discrete, DynSpace};
use rand_chacha::ChaCha8Rng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

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
    rng: ChaCha8Rng,
}

#[derive(Serialize, Deserialize)]
struct SquaredState {
    grid_flat: Vec<f32>,
    agent_pos: (usize, usize),
    targets: Vec<(usize, usize)>,
    tick: u32,
    rng: ChaCha8Rng,
}

// 8 movement directions: N, S, W, E, NE, NW, SE, SW
const MOVES: [(i32, i32); 8] = [
    (0, -1),
    (0, 1),
    (-1, 0),
    (1, 0),
    (1, -1),
    (-1, -1),
    (1, 1),
    (-1, 1),
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
            rng: ChaCha8Rng::from_entropy(),
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
            self.rng = ChaCha8Rng::seed_from_u64(s);
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
            self.grid.iter().copied().collect(),
        )
        .unwrap();

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
        let min_dist = self
            .targets
            .iter()
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
        let dist_from_origin = (nx as i32 - self.distance_to_target as i32)
            .abs()
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
            self.grid.iter().copied().collect(),
        )
        .unwrap();

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
                    line.push('🟢'); // Agent
                } else if val > 0.0 {
                    line.push('🎯'); // Target
                } else {
                    line.push('⬛'); // Empty
                }
            }
            lines.push(line);
        }
        Some(lines.join("\n"))
    }

    fn is_done(&self) -> bool {
        self.tick >= self.max_ticks
    }

    fn state(&self) -> Vec<u8> {
        let state = SquaredState {
            grid_flat: self.grid.iter().copied().collect(),
            agent_pos: self.agent_pos,
            targets: self.targets.clone(),
            tick: self.tick,
            rng: self.rng.clone(),
        };
        serde_json::to_vec(&state).unwrap()
    }

    fn set_state(&mut self, state: &[u8]) {
        let decoded: SquaredState = serde_json::from_slice(state).unwrap();
        self.grid = Array2::from_shape_vec((self.grid_size, self.grid_size), decoded.grid_flat).unwrap();
        self.agent_pos = decoded.agent_pos;
        self.targets = decoded.targets;
        self.tick = decoded.tick;
        self.rng = decoded.rng;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_squared_creation() {
        let env = Squared::new(2);
        assert_eq!(env.grid_size, 5); // 2*2+1 = 5
    }

    #[test]
    fn test_squared_reset() {
        let mut env = Squared::new(2);
        let (obs, _) = env.reset(Some(42));

        assert_eq!(obs.len(), 25); // 5x5 grid
    }

    #[test]
    fn test_squared_seed_consistency() {
        let mut env1 = Squared::new(3);
        let mut env2 = Squared::new(3);

        let (obs1, _) = env1.reset(Some(1234));
        let (obs2, _) = env2.reset(Some(1234));

        assert_eq!(obs1, obs2);
        assert_eq!(env1.targets, env2.targets);
    }

    #[test]
    fn test_squared_serialization() {
        let mut env = Squared::new(3);
        env.reset(Some(777));

        for _ in 0..5 {
            env.step(&ArrayD::from_elem(IxDyn(&[1]), 2.0)); // Move West
        }

        let state = env.state();
        let res1 = env.step(&ArrayD::from_elem(IxDyn(&[1]), 3.0)); // Move East

        env.set_state(&state);
        let res2 = env.step(&ArrayD::from_elem(IxDyn(&[1]), 3.0));

        assert_eq!(res1.observation, res2.observation);
    }
}
