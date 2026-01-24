use pufferlib::prelude::*;
use pufferlib_envs::CartPole;
use ndarray::Array2;

#[test]
fn test_vector_parity() {
    let num_envs = 4;
    let seed = 42;
    
    // Create serial backend
    let mut serial = VecEnv::from_backend(pufferlib::vector::Serial::new(
        || CartPole::new(),
        num_envs
    ));
    
    // Create parallel backend
    let mut parallel = VecEnv::from_backend(pufferlib::vector::Parallel::new(
        || CartPole::new(),
        num_envs
    ));
    
    // Reset both
    let (s_obs, _) = serial.reset(Some(seed));
    let (p_obs, _) = parallel.reset(Some(seed));
    
    // Initial observations must match
    assert_eq!(s_obs, p_obs, "Initial observations mismatched");
    
    // Step both for N steps with same actions
    let action_shape = serial.action_space().shape()[0];
    let actions = Array2::from_elem((num_envs, action_shape), 1.0); // CartPole action 1.0 = Right
    
    for _ in 0..10 {
        let s_res = serial.step(&actions);
        let p_res = parallel.step(&actions);
        
        assert_eq!(s_res.observations, p_res.observations, "Step observations mismatched");
        assert_eq!(s_res.rewards, p_res.rewards, "Step rewards mismatched");
        assert_eq!(s_res.terminated, p_res.terminated, "Step terminated mismatched");
    }
}
