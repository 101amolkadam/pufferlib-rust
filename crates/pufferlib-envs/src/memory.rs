//! Memory sequence environment.

use ndarray::{ArrayD, IxDyn};
use pufferlib::env::{EnvInfo, PufferEnv, StepResult};
use pufferlib::spaces::{Box as BoxSpace, Discrete, DynSpace};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Memory environment
///
/// The agent must remember and repeat a sequence after a delay.
/// Tests memory capacity and credit assignment.
pub struct Memory {
    /// Length of sequence to remember
    mem_length: usize,
    /// Delay between sequence presentation and reproduction
    mem_delay: usize,
    /// Total episode length
    horizon: usize,
    /// Solution sequence
    solution: Vec<f32>,
    /// Agent's submissions
    submission: Vec<f32>,
    /// Current step
    tick: usize,
    /// RNG
    rng: StdRng,
}

impl Memory {
    /// Create a new memory environment
    pub fn new(mem_length: usize, mem_delay: usize) -> Self {
        Self {
            mem_length,
            mem_delay,
            horizon: 2 * mem_length + mem_delay,
            solution: Vec::new(),
            submission: Vec::new(),
            tick: 0,
            rng: StdRng::from_entropy(),
        }
    }

    /// Create with default delay
    pub fn with_length(mem_length: usize) -> Self {
        Self::new(mem_length, 0)
    }
}

impl PufferEnv for Memory {
    fn observation_space(&self) -> DynSpace {
        DynSpace::Box(BoxSpace::uniform(&[1], -1.0, 1.0))
    }

    fn action_space(&self) -> DynSpace {
        DynSpace::Discrete(Discrete::new(2))
    }

    fn reset(&mut self, seed: Option<u64>) -> (ArrayD<f32>, EnvInfo) {
        if let Some(s) = seed {
            self.rng = StdRng::seed_from_u64(s);
        }

        // Generate random solution sequence
        self.solution = (0..self.horizon)
            .map(|i| {
                if i < self.mem_length {
                    if self.rng.gen::<bool>() {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    -1.0 // Placeholder
                }
            })
            .collect();

        self.submission = vec![-1.0; self.horizon];
        self.tick = 0;

        let obs = ArrayD::from_elem(IxDyn(&[1]), self.solution[0]);
        (obs, EnvInfo::new())
    }

    fn step(&mut self, action: &ArrayD<f32>) -> StepResult {
        let action_val = action.iter().next().unwrap().round();
        assert!(action_val == 0.0 || action_val == 1.0);

        let mut reward = 0.0;
        let mut ob = 0.0;

        // During sequence presentation
        if self.tick < self.mem_length {
            if self.tick + 1 < self.mem_length {
                ob = self.solution[self.tick + 1];
            }
            // Reward for outputting 0 during presentation
            if action_val == 0.0 {
                reward = 1.0;
            }
        }

        // During reproduction phase
        if self.tick >= self.mem_length + self.mem_delay {
            let idx = self.tick - self.mem_length - self.mem_delay;
            if idx < self.mem_length {
                let expected = self.solution[idx];
                reward = if action_val == expected { 1.0 } else { 0.0 };
                self.submission[self.tick] = action_val;
            }
        }

        self.tick += 1;
        let terminal = self.tick >= self.horizon;

        let info = if terminal {
            // Check if entire sequence matches
            let correct = (0..self.mem_length).all(|i| {
                let sub_idx = self.mem_length + self.mem_delay + i;
                self.submission.get(sub_idx).copied() == Some(self.solution[i])
            });
            EnvInfo::new().with_extra("score", if correct { 1.0 } else { 0.0 })
        } else {
            EnvInfo::new()
        };

        let obs = ArrayD::from_elem(IxDyn(&[1]), ob);

        StepResult {
            observation: obs,
            reward,
            terminated: terminal,
            truncated: false,
            info,
        }
    }

    fn render(&self) -> Option<String> {
        let solution_str: String = self
            .solution
            .iter()
            .take(self.mem_length)
            .map(|&v| if v == 1.0 { '1' } else { '0' })
            .collect();

        let submission_str: String = self
            .submission
            .iter()
            .skip(self.mem_length + self.mem_delay)
            .take(self.mem_length)
            .map(|&v| {
                if v == 1.0 {
                    '1'
                } else if v == 0.0 {
                    '0'
                } else {
                    '_'
                }
            })
            .collect();

        Some(format!(
            "Solution:   {}\nSubmission: {}\nStep: {}/{}",
            solution_str, submission_str, self.tick, self.horizon
        ))
    }

    fn is_done(&self) -> bool {
        self.tick >= self.horizon
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_creation() {
        let env = Memory::new(3, 2);
        assert_eq!(env.horizon, 8); // 2*3 + 2
    }

    #[test]
    fn test_memory_reset() {
        let mut env = Memory::new(3, 0);
        let (obs, _) = env.reset(Some(42));

        assert_eq!(obs.len(), 1);
    }

    #[test]
    fn test_memory_seed_consistency() {
        let mut env1 = Memory::new(5, 5);
        let mut env2 = Memory::new(5, 5);

        env1.reset(Some(999));
        env2.reset(Some(999));

        assert_eq!(env1.solution, env2.solution);
    }
}
