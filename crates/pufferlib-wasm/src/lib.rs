//! WebAssembly bindings for PufferLib.

use ndarray::{ArrayD, IxDyn};
use pufferlib::env::PufferEnv;
use pufferlib_envs::CartPole;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct PufferWasmEnv {
    env: CartPole,
}

#[wasm_bindgen]
pub struct StepResult {
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
    observation: Vec<f32>,
}

#[wasm_bindgen]
impl StepResult {
    #[wasm_bindgen(getter)]
    pub fn observation(&self) -> Vec<f32> {
        self.observation.clone()
    }
}

#[wasm_bindgen]
impl PufferWasmEnv {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            env: CartPole::new(),
        }
    }

    /// Reset the environment and return observations as a flat Float32Array
    pub fn reset(&mut self, seed: Option<u64>) -> Vec<f32> {
        let (obs, _) = self.env.reset(seed);
        obs.as_slice().unwrap().to_vec()
    }

    /// Step the environment and return a result object
    pub fn step(&mut self, action: f32) -> StepResult {
        let action_array = ArrayD::from_elem(IxDyn(&[1]), action);
        let result = self.env.step(&action_array);

        StepResult {
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            observation: result.observation.as_slice().unwrap().to_vec(),
        }
    }

    pub fn is_done(&self) -> bool {
        self.env.is_done()
    }
}

impl Default for PufferWasmEnv {
    fn default() -> Self {
        Self::new()
    }
}
