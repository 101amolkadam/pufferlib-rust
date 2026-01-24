//! Mock multi-agent environment for testing emulation layer features.

use ndarray::{ArrayD, IxDyn};
use pufferlib::env::{Action, EnvInfo, Observation, RawPufferEnv, RawStepResult};
use pufferlib::spaces::{Box as BoxSpace, Dict, DynSpace};
use std::collections::HashMap;

/// A simple multi-agent environment where agents move on a 1D line
pub struct MockMarl {
    num_agents: usize,
    agent_positions: Vec<f32>,
    tick: u32,
    max_ticks: u32,
}

impl MockMarl {
    pub fn new(num_agents: usize) -> Self {
        Self {
            num_agents,
            agent_positions: vec![0.0; num_agents],
            tick: 0,
            max_ticks: 10,
        }
    }
}

impl RawPufferEnv for MockMarl {
    fn observation_space(&self) -> DynSpace {
        // Each agent sees its position
        DynSpace::Box(BoxSpace::uniform(&[1], -10.0, 10.0))
    }

    fn action_space(&self) -> DynSpace {
        // Each agent can move left or right
        DynSpace::Box(BoxSpace::uniform(&[1], -1.0, 1.0))
    }

    fn reset(&mut self, _seed: Option<u64>) -> (Observation, EnvInfo) {
        self.agent_positions = vec![0.0; self.num_agents];
        self.tick = 0;

        // Return first agent's observation as a "default" reset for single-agent compatibility
        (
            Observation::Array(ArrayD::from_elem(IxDyn(&[1]), 0.0)),
            EnvInfo::new(),
        )
    }

    fn step(&mut self, action: &Action) -> RawStepResult {
        if let Action::Array(a) = action {
            let move_val = a.iter().next().unwrap();
            self.agent_positions[0] += move_val;
        }

        self.tick += 1;
        let done = self.tick >= self.max_ticks;

        RawStepResult {
            observation: Observation::Array(ArrayD::from_elem(
                IxDyn(&[1]),
                self.agent_positions[0],
            )),
            reward: 1.0,
            terminated: done,
            truncated: false,
            info: EnvInfo::new(),
        }
    }

    fn num_agents(&self) -> usize {
        self.num_agents
    }

    fn active_agents(&self) -> Vec<u32> {
        // For testing "variable population", only even agents are active
        (0..self.num_agents as u32).filter(|i| i % 2 == 0).collect()
    }

    fn multi_step(&mut self, actions: &HashMap<u32, Action>) -> HashMap<u32, RawStepResult> {
        let mut results = HashMap::new();
        self.tick += 1;
        let done = self.tick >= self.max_ticks;

        for &id in &self.active_agents() {
            if let Some(Action::Array(a)) = actions.get(&id) {
                let move_val = a.iter().next().unwrap();
                self.agent_positions[id as usize] += move_val;
            }

            results.insert(
                id,
                RawStepResult {
                    observation: Observation::Array(ArrayD::from_elem(
                        IxDyn(&[1]),
                        self.agent_positions[id as usize],
                    )),
                    reward: 1.0,
                    terminated: done,
                    truncated: false,
                    info: EnvInfo::new(),
                },
            );
        }

        results
    }
}
