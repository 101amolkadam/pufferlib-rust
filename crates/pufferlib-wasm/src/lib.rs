//! WebAssembly bindings for PufferLib.

use wasm_bindgen::prelude::*;
use pufferlib::env::{PufferEnv, StepResult};
use ndarray::ArrayD;

#[wasm_bindgen]
pub struct PufferWasmEnv {
    // Note: We can't store Box<dyn PufferEnv> directly in a #[wasm_bindgen] struct
    // if we want to expose it to JS easily. We'll use a concrete type or a wrapper.
    // For now, this is a placeholder for the WASM environment bridge.
}

#[wasm_bindgen]
impl PufferWasmEnv {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {}
    }

    /// Reset the environment and return observations as a flat Float32Array
    pub fn reset(&mut self, _seed: Option<u64>) -> Vec<f32> {
        // Placeholder for actual reset logic
        vec![0.0; 1]
    }

    /// Step the environment and return rewards/done
    pub fn step(&mut self, _action: Vec<f32>) -> JsValue {
        // Placeholder for actual step logic
        JsValue::from_f64(0.0)
    }
}
