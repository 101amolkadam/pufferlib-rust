use pufferlib::env::{Action, EmulationLayer, Observation, PufferEnv, RawPufferEnv, RawStepResult};
use pufferlib::spaces::{Box as BoxSpace, Dict, DynSpace, Tuple};
use pufferlib_envs::MockMarl;
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

#[test]
fn test_recursive_flattening() {
    // Manually test flattening of a nested structure using EmulationLayer logic
    // (Actually testing SpaceTree via EmulationLayer)
    
    struct ComplexEnv;
    impl RawPufferEnv for ComplexEnv {
        fn observation_space(&self) -> DynSpace {
            let mut dict = HashMap::new();
            dict.insert("a".to_string(), DynSpace::Box(BoxSpace::uniform(&[2], 0.0, 1.0)));
            dict.insert("b".to_string(), DynSpace::Tuple(Tuple::new(vec![
                DynSpace::Box(BoxSpace::uniform(&[1], 0.0, 1.0))
            ])));
            DynSpace::Dict(Dict::new(dict))
        }
        fn action_space(&self) -> DynSpace {
            DynSpace::Box(BoxSpace::uniform(&[1], 0.0, 1.0))
        }
        fn reset(&mut self, _seed: Option<u64>) -> (Observation, pufferlib::env::EnvInfo) {
            let mut map = HashMap::new();
            map.insert("a".to_string(), Observation::Array(ArrayD::from_elem(IxDyn(&[2]), 0.5)));
            map.insert("b".to_string(), Observation::Tuple(vec![
                Observation::Array(ArrayD::from_elem(IxDyn(&[1]), 1.0))
            ]));
            (Observation::Dict(map), pufferlib::env::EnvInfo::new())
        }
        fn step(&mut self, _action: &Action) -> RawStepResult {
            unimplemented!()
        }
    }

    let mut emulated = EmulationLayer::new(ComplexEnv);
    let (obs, _) = emulated.reset(None);
    
    // Total size: 2 (a) + 1 (b[0]) = 3
    assert_eq!(obs.len(), 3);
    assert_eq!(obs[0], 0.5);
    assert_eq!(obs[1], 0.5);
    assert_eq!(obs[2], 1.0);
}

#[test]
fn test_marl_emulation() {
    let mock = MockMarl::new(4); // 4 agents total, only even expected active
    let mut emulated = EmulationLayer::new(mock);
    
    // Reset should return combined observations [4, 1]
    let (obs, _) = emulated.reset(None);
    assert_eq!(obs.shape(), &[4, 1]);
    
    // Step with stacked actions
    let actions = ArrayD::from_elem(IxDyn(&[4, 1]), 1.0);
    let res = emulated.step(&actions);
    
    // Observations should reflect movement for active agents, 0 for inactive/padded
    assert_eq!(res.observation.shape(), &[4, 1]);
    let obs_data = res.observation.as_slice().unwrap();
    assert_eq!(obs_data[0], 1.0); // Agent 0 active
    assert_eq!(obs_data[1], 0.0); // Agent 1 inactive (padded)
    assert_eq!(obs_data[2], 1.0); // Agent 2 active
    assert_eq!(obs_data[3], 0.0); // Agent 3 inactive (padded)
}
