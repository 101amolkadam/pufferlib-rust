//! Mock environment with heterogeneous agents for testing emulation layer.

use ndarray::{ArrayD, IxDyn};
use pufferlib::env::{Action, EnvInfo, Observation, RawPufferEnv, RawStepResult};
use pufferlib::spaces::{Box as BoxSpace, Dict, DynSpace};
use std::collections::HashMap;

/// A mock environment where two agents have different observation/action spaces.
/// Agent 0: "Small" (Box observation)
/// Agent 1: "Large" (Dict observation)
pub struct HeteroMock {
    pub tick: u32,
}

impl HeteroMock {
    pub fn new() -> Self {
        Self { tick: 0 }
    }
}

impl RawPufferEnv for HeteroMock {
    fn observation_space(&self) -> DynSpace {
        // This is a dummy for heterogeneous case
        DynSpace::Box(BoxSpace::uniform(&[1], 0.0, 1.0))
    }

    fn action_space(&self) -> DynSpace {
        DynSpace::Box(BoxSpace::uniform(&[1], 0.0, 1.0))
    }

    fn agent_type(&self, agent_id: u32) -> String {
        if agent_id == 0 {
            "small".to_string()
        } else {
            "large".to_string()
        }
    }

    fn observation_space_for(&self, agent_type: &str) -> DynSpace {
        match agent_type {
            "small" => DynSpace::Box(BoxSpace::uniform(&[2], 0.0, 1.0)),
            "large" => {
                let mut dict = HashMap::new();
                dict.insert(
                    "a".to_string(),
                    DynSpace::Box(BoxSpace::uniform(&[4], 0.0, 1.0)),
                );
                DynSpace::Dict(Dict::new(dict))
            }
            _ => panic!("Unknown agent type"),
        }
    }

    fn action_space_for(&self, agent_type: &str) -> DynSpace {
        match agent_type {
            "small" => DynSpace::Box(BoxSpace::uniform(&[1], 0.0, 1.0)),
            "large" => DynSpace::Box(BoxSpace::uniform(&[1], 0.0, 1.0)),
            _ => panic!("Unknown agent type"),
        }
    }

    fn reset(&mut self, _seed: Option<u64>) -> (Observation, EnvInfo) {
        self.tick = 0;
        // Return dummy observation for default reset
        (
            Observation::Array(ArrayD::from_elem(IxDyn(&[2]), 0.0)),
            EnvInfo::new(),
        )
    }

    fn step(&mut self, _action: &Action) -> RawStepResult {
        unimplemented!("multi_step preferred")
    }

    fn num_agents(&self) -> usize {
        2
    }

    fn active_agents(&self) -> Vec<u32> {
        vec![0, 1]
    }

    fn multi_step(&mut self, _actions: &HashMap<u32, Action>) -> HashMap<u32, RawStepResult> {
        let mut results = HashMap::new();

        // Agent 0 (Small)
        results.insert(
            0,
            RawStepResult {
                observation: Observation::Array(ArrayD::from_elem(IxDyn(&[2]), 0.5)),
                reward: 1.0,
                terminated: false,
                truncated: false,
                info: EnvInfo::new(),
            },
        );

        // Agent 1 (Large)
        let mut map = HashMap::new();
        map.insert(
            "a".to_string(),
            Observation::Array(ArrayD::from_elem(IxDyn(&[4]), 1.0)),
        );
        results.insert(
            1,
            RawStepResult {
                observation: Observation::Dict(map),
                reward: 2.0,
                terminated: false,
                truncated: false,
                info: EnvInfo::new(),
            },
        );

        results
    }
}
