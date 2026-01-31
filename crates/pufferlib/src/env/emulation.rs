use super::{EnvInfo, Observation, PufferEnv, RawPufferEnv, StepResult};
use crate::spaces::{DynSpace, SpaceTree};
use crate::types::{vec, HashMap, String, Vec};
use ndarray::{ArrayD, IxDyn};

/// Emulation layer that wraps a RawPufferEnv and provides flattening, padding, and masking.
pub struct EmulationLayer<E: RawPufferEnv> {
    /// Inner raw environment
    env: E,
    /// Maximum number of agents
    num_agents: usize,
    /// Metadata for observation/action flattening per agent type
    agent_trees: HashMap<String, (SpaceTree, SpaceTree)>,
    /// Maximum observation size across all agent types
    max_obs_size: usize,
}

impl<E: RawPufferEnv> EmulationLayer<E> {
    /// Create a new emulation layer
    pub fn new(env: E) -> Self {
        let num_agents = env.num_agents();
        let mut agent_trees = HashMap::new();
        let mut max_obs_size = 0;

        // Discover all agent types and their spaces
        for i in 0..num_agents as u32 {
            let agent_type = env.agent_type(i);
            let trees = agent_trees.entry(agent_type.clone()).or_insert_with(|| {
                let obs_space = env.observation_space_for(&agent_type);
                let act_space = env.action_space_for(&agent_type);

                let obs_tree = SpaceTree::from_space(&obs_space);
                let act_tree = SpaceTree::from_space(&act_space);

                (obs_tree, act_tree)
            });
            max_obs_size = max_obs_size.max(trees.0.size());
        }

        Self {
            env,
            num_agents,
            agent_trees,
            max_obs_size,
        }
    }

    fn flatten_observation(&self, obs: &Observation, agent_type: &str) -> ArrayD<f32> {
        let (obs_tree, _) = self
            .agent_trees
            .get(agent_type)
            .expect("Unknown agent type");
        let mut buf = vec![0.0; self.max_obs_size];
        obs_tree.flatten(obs, &mut buf);
        ArrayD::from_shape_vec(IxDyn(&[self.max_obs_size]), buf).unwrap()
    }

    /// Reset the environment
    pub fn reset(&mut self, seed: Option<u64>) -> (ArrayD<f32>, EnvInfo) {
        let (obs, info) = self.env.reset(seed);
        let agent_type = self.env.agent_type(0);
        (self.flatten_observation(&obs, &agent_type), info)
    }
}

impl<E: RawPufferEnv> PufferEnv for EmulationLayer<E> {
    fn observation_space(&self) -> DynSpace {
        // Heterogeneous: return a unified Box space for the flat tensor
        DynSpace::Box(crate::spaces::Box::uniform(
            &[self.max_obs_size],
            -f32::INFINITY,
            f32::INFINITY,
        ))
    }

    fn action_space(&self) -> DynSpace {
        // Return representative action space (first agent)
        self.env.action_space_for(&self.env.agent_type(0))
    }

    fn reset(&mut self, seed: Option<u64>) -> (ArrayD<f32>, EnvInfo) {
        let (obs, info) = self.env.reset(seed);
        let agent_type = self.env.agent_type(0);
        (self.flatten_observation(&obs, &agent_type), info)
    }

    fn step(&mut self, action: &ArrayD<f32>) -> StepResult {
        if self.num_agents == 1 {
            let agent_type = self.env.agent_type(0);
            let (_, action_tree) = self
                .agent_trees
                .get(&agent_type)
                .expect("Unknown agent type");
            let structured_action = action_tree.unflatten(action.as_slice().unwrap());
            let res = self.env.step(&structured_action);
            return StepResult {
                observation: self.flatten_observation(&res.observation, &agent_type),
                reward: res.reward,
                terminated: res.terminated,
                truncated: res.truncated,
                info: res.info,
                cost: res.cost,
            };
        }

        // Multi-agent case: action is [num_agents, action_size]
        let mut structured_actions = HashMap::new();

        for i in 0..self.num_agents as u32 {
            let agent_type = self.env.agent_type(i);
            let (_, action_tree) = self
                .agent_trees
                .get(&agent_type)
                .expect("Unknown agent type");
            let action_size = action_tree.size();

            let start = i as usize * action_size;
            let slice = action.as_slice().unwrap();
            let agent_action = action_tree.unflatten(&slice[start..start + action_size]);
            structured_actions.insert(i, agent_action);
        }

        let res_map = self.env.multi_step(&structured_actions);

        let mut combined_obs = ArrayD::from_elem(IxDyn(&[self.num_agents, self.max_obs_size]), 0.0);
        let mut total_reward = 0.0;
        let mut terminated = false;
        let mut truncated = false;
        let mut combined_info = EnvInfo::new();
        let mut total_cost = 0.0;

        for i in 0..self.num_agents as u32 {
            if let Some(res) = res_map.get(&i) {
                let agent_type = self.env.agent_type(i);
                let mut slice = combined_obs.slice_mut(ndarray::s![i as usize, ..]);
                let flat = self.flatten_observation(&res.observation, &agent_type);
                slice.assign(&flat.view().to_shape((self.max_obs_size,)).unwrap());

                total_reward += res.reward;
                terminated |= res.terminated;
                truncated |= res.truncated;
                if i == 0 {
                    combined_info = res.info.clone();
                }
                total_cost += res.cost;
            }
        }

        StepResult {
            observation: combined_obs,
            reward: total_reward,
            terminated,
            truncated,
            info: combined_info,
            cost: total_cost,
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

    fn state(&self) -> Vec<u8> {
        self.env.state()
    }

    fn set_state(&mut self, state: &[u8]) {
        self.env.set_state(state);
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
                        cost: 0.0,
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
        };
        let mut emulated = EmulationLayer::new(mock);
        let actions = ArrayD::from_elem(IxDyn(&[4, 1]), 1.0);
        let res = emulated.step(&actions);
        assert_eq!(res.observation.shape(), &[4, 1]);
        let obs_data = res.observation.as_slice().unwrap();
        assert_eq!(obs_data[0], 1.0);
        assert_eq!(obs_data[1], 0.0);
        assert_eq!(obs_data[2], 1.0);
        assert_eq!(obs_data[3], 0.0);
    }

    struct HeteroMock;
    impl RawPufferEnv for HeteroMock {
        fn observation_space(&self) -> DynSpace {
            DynSpace::Box(BoxSpace::uniform(&[1], 0.0, 1.0))
        }
        fn action_space(&self) -> DynSpace {
            DynSpace::Box(BoxSpace::uniform(&[1], 0.0, 1.0))
        }
        fn num_agents(&self) -> usize {
            2
        }
        fn active_agents(&self) -> Vec<u32> {
            vec![0, 1]
        }
        fn agent_type(&self, id: u32) -> String {
            if id == 0 {
                "s".to_string()
            } else {
                "l".to_string()
            }
        }
        fn observation_space_for(&self, t: &str) -> DynSpace {
            if t == "s" {
                DynSpace::Box(BoxSpace::uniform(&[2], 0.0, 1.0))
            } else {
                let mut d = HashMap::new();
                d.insert(
                    "a".to_string(),
                    DynSpace::Box(BoxSpace::uniform(&[4], 0.0, 1.0)),
                );
                DynSpace::Dict(Dict::new(d))
            }
        }
        fn reset(&mut self, _s: Option<u64>) -> (Observation, EnvInfo) {
            (
                Observation::Array(ArrayD::from_elem(IxDyn(&[2]), 0.0)),
                EnvInfo::new(),
            )
        }
        fn step(&mut self, _a: &Action) -> RawStepResult {
            unimplemented!()
        }
        fn multi_step(&mut self, _a: &HashMap<u32, Action>) -> HashMap<u32, RawStepResult> {
            let mut r = HashMap::new();
            r.insert(
                0,
                RawStepResult {
                    observation: Observation::Array(ArrayD::from_elem(IxDyn(&[2]), 0.5)),
                    reward: 1.0,
                    terminated: false,
                    truncated: false,
                    info: EnvInfo::new(),
                    cost: 0.0,
                },
            );
            let mut m = HashMap::new();
            m.insert(
                "a".to_string(),
                Observation::Array(ArrayD::from_elem(IxDyn(&[4]), 1.0)),
            );
            r.insert(
                1,
                RawStepResult {
                    observation: Observation::Dict(m),
                    reward: 2.0,
                    terminated: false,
                    truncated: false,
                    info: EnvInfo::new(),
                    cost: 0.0,
                },
            );
            r
        }
    }

    #[test]
    fn test_heterogeneous_emulation() {
        let mut emulated = EmulationLayer::new(HeteroMock);
        let actions = ArrayD::from_elem(IxDyn(&[2, 1]), 0.0);
        let res = emulated.step(&actions);
        assert_eq!(res.observation.shape(), &[2, 4]);
        let obs = res.observation.as_slice().unwrap();
        assert_eq!(obs[0], 0.5);
        assert_eq!(obs[1], 0.5);
        assert_eq!(obs[2], 0.0);
        assert_eq!(obs[3], 0.0); // Padded agent 0
        assert_eq!(obs[4], 1.0);
        assert_eq!(obs[5], 1.0);
        assert_eq!(obs[6], 1.0);
        assert_eq!(obs[7], 1.0); // Agent 1
    }
}
