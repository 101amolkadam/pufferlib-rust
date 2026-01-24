use pufferlib::env::{PufferEnv, StepResult};
use pufferlib_envs::{CartPole, Memory, Squared};
use ndarray::{ArrayD, IxDyn};

#[test]
fn test_cartpole_serialization() {
    let mut env = CartPole::new();
    env.reset(Some(42));
    
    // Take some steps
    for _ in 0..10 {
        env.step(&ArrayD::from_elem(IxDyn(&[1]), 1.0));
    }
    
    let state = env.state();
    
    // Step further
    let res1 = env.step(&ArrayD::from_elem(IxDyn(&[1]), 0.0));
    let next_state = env.state();
    
    // Restore and re-step
    env.set_state(&state);
    let res2 = env.step(&ArrayD::from_elem(IxDyn(&[1]), 0.0));
    
    assert_eq!(res1.observation, res2.observation);
    assert_eq!(res1.reward, res2.reward);
    assert_eq!(env.state(), next_state);
}

#[test]
fn test_memory_serialization() {
    let mut env = Memory::new(5, 2);
    env.reset(Some(123));
    
    for _ in 0..3 {
        env.step(&ArrayD::from_elem(IxDyn(&[1]), 0.0));
    }
    
    let state = env.state();
    let res1 = env.step(&ArrayD::from_elem(IxDyn(&[1]), 1.0));
    
    env.set_state(&state);
    let res2 = env.step(&ArrayD::from_elem(IxDyn(&[1]), 1.0));
    
    assert_eq!(res1.observation, res2.observation);
    assert_eq!(res1.reward, res2.reward);
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
