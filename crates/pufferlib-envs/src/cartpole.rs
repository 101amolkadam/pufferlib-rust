//! CartPole classic control environment.

use ndarray::{ArrayD, IxDyn};
use pufferlib::env::{PufferEnv, EnvInfo, StepResult};
use pufferlib::spaces::{DynSpace, Discrete, Box as BoxSpace};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::f32::consts::PI;

/// CartPole environment
///
/// A pole is attached to a cart on a frictionless track. The goal
/// is to balance the pole by applying forces to the cart.
///
/// Observation: [cart_pos, cart_vel, pole_angle, pole_vel]
/// Action: 0 = push left, 1 = push right
pub struct CartPole {
    // Physics constants
    gravity: f32,
    _mass_cart: f32,
    mass_pole: f32,
    total_mass: f32,
    length: f32,  // half-pole length
    pole_mass_length: f32,
    force_mag: f32,
    tau: f32,  // timestep
    
    // Thresholds
    theta_threshold: f32,
    x_threshold: f32,
    max_steps: u32,
    
    // State
    state: [f32; 4],  // x, x_dot, theta, theta_dot
    steps: u32,
    done: bool,
    rng: StdRng,
}

impl CartPole {
    /// Create a new CartPole environment
    pub fn new() -> Self {
        let mass_cart = 1.0;
        let mass_pole = 0.1;
        let length = 0.5;
        
        Self {
            gravity: 9.8,
            _mass_cart: mass_cart,
            mass_pole,
            total_mass: mass_cart + mass_pole,
            length,
            pole_mass_length: mass_pole * length,
            force_mag: 10.0,
            tau: 0.02,
            theta_threshold: 12.0 * 2.0 * PI / 360.0,  // 12 degrees
            x_threshold: 2.4,
            max_steps: 500,
            state: [0.0; 4],
            steps: 0,
            done: false,
            rng: StdRng::from_entropy(),
        }
    }
    
    fn is_terminal(&self) -> bool {
        let x = self.state[0];
        let theta = self.state[2];
        
        x.abs() > self.x_threshold || theta.abs() > self.theta_threshold
    }
}

impl Default for CartPole {
    fn default() -> Self {
        Self::new()
    }
}

impl PufferEnv for CartPole {
    fn observation_space(&self) -> DynSpace {
        // Observation bounds (loosely)
        DynSpace::Box(BoxSpace::uniform(&[4], -4.8, 4.8))
    }
    
    fn action_space(&self) -> DynSpace {
        DynSpace::Discrete(Discrete::new(2))
    }
    
    fn reset(&mut self, seed: Option<u64>) -> (ArrayD<f32>, EnvInfo) {
        if let Some(s) = seed {
            self.rng = StdRng::seed_from_u64(s);
        }
        
        // Initialize state randomly in [-0.05, 0.05]
        for i in 0..4 {
            self.state[i] = self.rng.gen::<f32>() * 0.1 - 0.05;
        }
        
        self.steps = 0;
        self.done = false;
        
        let obs = ArrayD::from_shape_vec(IxDyn(&[4]), self.state.to_vec()).unwrap();
        (obs, EnvInfo::new())
    }
    
    fn step(&mut self, action: &ArrayD<f32>) -> StepResult {
        let action_idx = action.iter().next().unwrap().round() as usize;
        assert!(action_idx < 2);
        
        let [x, x_dot, theta, theta_dot] = self.state;
        
        // Apply force
        let force = if action_idx == 1 { self.force_mag } else { -self.force_mag };
        
        // Physics simulation
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        
        let temp = (force + self.pole_mass_length * theta_dot * theta_dot * sin_theta) / self.total_mass;
        let theta_acc = (self.gravity * sin_theta - cos_theta * temp) 
            / (self.length * (4.0 / 3.0 - self.mass_pole * cos_theta * cos_theta / self.total_mass));
        let x_acc = temp - self.pole_mass_length * theta_acc * cos_theta / self.total_mass;
        
        // Euler integration
        self.state[0] = x + self.tau * x_dot;
        self.state[1] = x_dot + self.tau * x_acc;
        self.state[2] = theta + self.tau * theta_dot;
        self.state[3] = theta_dot + self.tau * theta_acc;
        
        self.steps += 1;
        
        let terminated = self.is_terminal();
        let truncated = self.steps >= self.max_steps;
        self.done = terminated || truncated;
        
        let reward = if !terminated { 1.0 } else { 0.0 };
        
        let obs = ArrayD::from_shape_vec(IxDyn(&[4]), self.state.to_vec()).unwrap();
        
        let mut info = EnvInfo::new();
        if self.done {
            // Note: EpisodeStats wrapper handles return/length usually, 
            // but if we want it here:
            info = info.with_extra("episode_length", self.steps as f32);
        }
        
        StepResult {
            observation: obs,
            reward,
            terminated,
            truncated,
            info,
        }
    }
    
    fn render(&self) -> Option<String> {
        let [x, _, theta, _] = self.state;
        
        // Simple ASCII rendering
        let cart_pos = ((x + 2.4) / 4.8 * 20.0) as i32;
        let cart_pos = cart_pos.clamp(0, 20);
        
        let mut line = vec![' '; 21];
        line[cart_pos as usize] = if theta.abs() < 0.1 { '|' } else { '/' };
        
        Some(format!("[{}]", line.iter().collect::<String>()))
    }
    
    fn is_done(&self) -> bool {
        self.done
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cartpole_creation() {
        let env = CartPole::new();
        assert!(!env.is_done());
    }
    
    #[test]
    fn test_cartpole_reset() {
        let mut env = CartPole::new();
        let (obs, _) = env.reset(Some(42));
        
        assert_eq!(obs.len(), 4);
        assert!(!env.is_done());
    }
    
    #[test]
    fn test_cartpole_step() {
        let mut env = CartPole::new();
        env.reset(Some(42));
        
        let action = ArrayD::from_elem(IxDyn(&[1]), 1.0);
        let result = env.step(&action);
        
        assert_eq!(result.observation.len(), 4);
        assert_eq!(result.reward, 1.0);  // Should get reward if not terminated
    }
    
    #[test]
    fn test_cartpole_determinism() {
        let mut env1 = CartPole::new();
        let mut env2 = CartPole::new();
        
        env1.reset(Some(42));
        env2.reset(Some(42));
        
        for _ in 0..10 {
            let action = ArrayD::from_elem(IxDyn(&[1]), 1.0);
            let res1 = env1.step(&action);
            let res2 = env2.step(&action);
            assert_eq!(res1.observation, res2.observation);
        }
    }
}
