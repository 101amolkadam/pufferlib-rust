use super::{EnvInfo, Observation, PufferEnv, RawPufferEnv, StepResult};
use crate::spaces::{DynSpace, SpaceTree};
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

/// Emulation layer that wraps a RawPufferEnv and provides flattening, padding, and masking.
pub struct EmulationLayer<E: RawPufferEnv> {
    /// Inner raw environment
    env: E,
    /// Maximum number of agents
    num_agents: usize,
    /// Metadata for observation flattening
    obs_tree: SpaceTree,
    /// Metadata for action unflattening
    action_tree: SpaceTree,
}

impl<E: RawPufferEnv> EmulationLayer<E> {
    /// Create a new emulation layer
    pub fn new(env: E) -> Self {
        let max_agents = env.num_agents();
        let obs_space = env.observation_space();
        let action_space = env.action_space();

        let obs_tree = SpaceTree::from_space(&obs_space);
        let action_tree = SpaceTree::from_space(&action_space);

        Self {
            env,
            num_agents: max_agents,
            obs_tree,
            action_tree,
        }
    }

    fn flatten_observation(&self, obs: &Observation) -> ArrayD<f32> {
        let mut buf = vec![0.0; self.obs_tree.size()];
        self.obs_tree.flatten(obs, &mut buf);
        ArrayD::from_shape_vec(IxDyn(&[buf.len()]), buf).unwrap()
    }

    /// Reset the environment
    pub fn reset(&mut self, seed: Option<u64>) -> (ArrayD<f32>, EnvInfo) {
        let (obs, info) = self.env.reset(seed);
        (self.flatten_observation(&obs), info)
    }
}

impl<E: RawPufferEnv> PufferEnv for EmulationLayer<E> {
    fn observation_space(&self) -> DynSpace {
        self.env.observation_space()
    }

    fn action_space(&self) -> DynSpace {
        self.env.action_space()
    }

    fn reset(&mut self, seed: Option<u64>) -> (ArrayD<f32>, EnvInfo) {
        let (obs, info) = self.env.reset(seed);
        (self.flatten_observation(&obs), info)
    }

    fn step(&mut self, action: &ArrayD<f32>) -> StepResult {
        if self.num_agents == 1 {
            let structured_action = self.action_tree.unflatten(action.as_slice().unwrap());
            let res = self.env.step(&structured_action);
            return StepResult {
                observation: self.flatten_observation(&res.observation),
                reward: res.reward,
                terminated: res.terminated,
                truncated: res.truncated,
                info: res.info,
            };
        }

        // Multi-agent case: action is [num_agents, action_size]
        let action_size = self.action_tree.size();
        let mut structured_actions = HashMap::new();

        for i in 0..self.num_agents {
            let start = i * action_size;
            let slice = action.as_slice().unwrap();
            let agent_action = self
                .action_tree
                .unflatten(&slice[start..start + action_size]);
            structured_actions.insert(i as u32, agent_action);
        }

        let res_map = self.env.multi_step(&structured_actions);

        let flat_obs_size = self.obs_tree.size();
        let mut combined_obs = ArrayD::from_elem(IxDyn(&[self.num_agents, flat_obs_size]), 0.0);
        let mut total_reward = 0.0;
        let mut terminated = false;
        let mut truncated = false;
        let mut combined_info = EnvInfo::new();

        for i in 0..self.num_agents as u32 {
            if let Some(res) = res_map.get(&i) {
                let mut slice = combined_obs.slice_mut(ndarray::s![i as usize, ..]);
                let flat = self.flatten_observation(&res.observation);
                slice.assign(&flat.view().to_shape((flat_obs_size,)).unwrap());

                total_reward += res.reward;
                terminated |= res.terminated;
                truncated |= res.truncated;
                // Merge info if needed (simplified for now)
                if i == 0 {
                    combined_info = res.info.clone();
                }
            }
        }

        StepResult {
            observation: combined_obs,
            reward: total_reward,
            terminated,
            truncated,
            info: combined_info,
        }
    }

    fn render(&self) -> Option<String> {
        self.env.render()
    }

    fn close(&mut self) {
        self.env.close();
    }

    fn num_agents(&self) -> usize {
        self.num_agents
    }

    fn active_agents(&self) -> Vec<u32> {
        self.env.active_agents()
    }

    fn is_done(&self) -> bool {
        self.env.is_done()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::{Action, EnvInfo, Observation, RawPufferEnv, RawStepResult};
    use crate::spaces::{Box as BoxSpace, Dict, Tuple};

    #[test]
    fn test_recursive_flattening() {
        struct ComplexEnv;
        impl RawPufferEnv for ComplexEnv {
            fn observation_space(&self) -> DynSpace {
                let mut dict = HashMap::new();
                dict.insert(
                    "a".to_string(),
                    DynSpace::Box(BoxSpace::uniform(&[2], 0.0, 1.0)),
                );
                dict.insert(
                    "b".to_string(),
                    DynSpace::Tuple(Tuple::new(vec![DynSpace::Box(BoxSpace::uniform(
                        &[1],
                        0.0,
                        1.0,
                    ))])),
                );
                DynSpace::Dict(Dict::new(dict))
            }
            fn action_space(&self) -> DynSpace {
                DynSpace::Box(BoxSpace::uniform(&[1], 0.0, 1.0))
            }
            fn reset(&mut self, _seed: Option<u64>) -> (Observation, EnvInfo) {
                let mut map = HashMap::new();
                map.insert(
                    "a".to_string(),
                    Observation::Array(ArrayD::from_elem(IxDyn(&[2]), 0.5)),
                );
                map.insert(
                    "b".to_string(),
                    Observation::Tuple(vec![Observation::Array(ArrayD::from_elem(
                        IxDyn(&[1]),
                        1.0,
                    ))]),
                );
                (Observation::Dict(map), EnvInfo::new())
            }
            fn step(&mut self, _action: &Action) -> RawStepResult {
                unimplemented!()
            }
        }

        let mut emulated = EmulationLayer::new(ComplexEnv);
        let (obs, _) = emulated.reset(None);

        assert_eq!(obs.len(), 3);
        assert_eq!(obs[0], 0.5);
        assert_eq!(obs[1], 0.5);
        assert_eq!(obs[2], 1.0);
    }

    struct MockMarl {
        num_agents: usize,
        agent_positions: Vec<f32>,
        tick: u32,
    }

    impl RawPufferEnv for MockMarl {
        fn observation_space(&self) -> DynSpace {
            DynSpace::Box(BoxSpace::uniform(&[1], -10.0, 10.0))
        }
        fn action_space(&self) -> DynSpace {
            DynSpace::Box(BoxSpace::uniform(&[1], -1.0, 1.0))
        }
        fn reset(&mut self, _seed: Option<u64>) -> (Observation, EnvInfo) {
            self.agent_positions = vec![0.0; self.num_agents];
            self.tick = 0;
            (
                Observation::Array(ArrayD::from_elem(IxDyn(&[1]), 0.0)),
                EnvInfo::new(),
            )
        }
        fn step(&mut self, _action: &Action) -> RawStepResult {
            unimplemented!()
        }
        fn num_agents(&self) -> usize {
            self.num_agents
        }
        fn active_agents(&self) -> Vec<u32> {
            (0..self.num_agents as u32).filter(|i| i % 2 == 0).collect()
        }
        fn multi_step(&mut self, actions: &HashMap<u32, Action>) -> HashMap<u32, RawStepResult> {
            let mut results = HashMap::new();
            for &id in &self.active_agents() {
                if let Some(Action::Array(a)) = actions.get(&id) {
                    self.agent_positions[id as usize] += a[0];
                }
                results.insert(
                    id,
                    RawStepResult {
                        observation: Observation::Array(ArrayD::from_elem(
                            IxDyn(&[1]),
                            self.agent_positions[id as usize],
                        )),
                        reward: 1.0,
                        terminated: false,
                        truncated: false,
                        info: EnvInfo::new(),
                    },
                );
            }
            results
        }
    }

    #[test]
    fn test_marl_emulation() {
        let mock = MockMarl {
            num_agents: 4,
            agent_positions: vec![0.0; 4],
            tick: 0,
        };
        let mut emulated = EmulationLayer::new(mock);

        let actions = ArrayD::from_elem(IxDyn(&[4, 1]), 1.0);
        let res = emulated.step(&actions);

        assert_eq!(res.observation.shape(), &[4, 1]);
        let obs_data = res.observation.as_slice().unwrap();
        assert_eq!(obs_data[0], 1.0); // Agent 0 active
        assert_eq!(obs_data[1], 0.0); // Agent 1 inactive (padded)
        assert_eq!(obs_data[2], 1.0); // Agent 2 active
        assert_eq!(obs_data[3], 0.0); // Agent 3 inactive (padded)
    }
}
