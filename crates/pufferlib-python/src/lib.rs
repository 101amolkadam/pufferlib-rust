//! PyO3 bindings for PufferLib.
#![allow(clippy::useless_conversion)]

use ndarray::ArrayD;
use numpy::{PyArrayDyn, PyArrayMethods, ToPyArray};
use pufferlib::env::{PufferEnv, StepResult};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[cfg(feature = "torch")]
use pufferlib::mappo::MappoConfig;

/// Python wrapper for a PufferLib environment
#[pyclass]
pub struct PufferPythonEnv {
    pub env: Box<dyn PufferEnv>,
}

/// Type alias for step result to reduce complexity
type StepResultTuple = (Py<PyArrayDyn<f32>>, f32, bool, bool, PyObject);

#[pymethods]
impl PufferPythonEnv {
    /// Reset the environment
    #[pyo3(signature = (seed=None))]
    fn reset(
        &mut self,
        py: Python,
        seed: Option<u64>,
    ) -> PyResult<(Py<PyArrayDyn<f32>>, PyObject)> {
        let (obs, _info) = self.env.reset(seed);
        let py_obs = obs.to_pyarray_bound(py).unbind();
        let py_info = PyDict::new_bound(py).into_any().unbind();
        Ok((py_obs, py_info))
    }

    /// Step the environment
    #[allow(clippy::type_complexity)]
    fn step(&mut self, py: Python, action: Py<PyArrayDyn<f32>>) -> PyResult<StepResultTuple> {
        // Convert Python array to ndarray
        let action_bound = action.bind(py);
        let action_array: ArrayD<f32> = unsafe { action_bound.as_array().to_owned() };

        let result: StepResult = self.env.step(&action_array);

        let py_obs = result.observation.to_pyarray_bound(py).unbind();
        let py_info = PyDict::new_bound(py).into_any().unbind();

        Ok((
            py_obs,
            result.reward,
            result.terminated,
            result.truncated,
            py_info,
        ))
    }
}

// --- MAPPO Bindings ---

#[cfg(feature = "torch")]
#[pyclass]
#[derive(Clone)]
struct PyMappoConfig {
    #[pyo3(get, set)]
    pub num_agents: usize,
    #[pyo3(get, set)]
    pub obs_dim: i64,
    #[pyo3(get, set)]
    pub global_state_dim: i64,
    #[pyo3(get, set)]
    pub hidden_dim: i64,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub gamma: f64,
    #[pyo3(get, set)]
    pub gae_lambda: f64,
    #[pyo3(get, set)]
    pub clip_coef: f64,
    #[pyo3(get, set)]
    pub update_epochs: usize,
    #[pyo3(get, set)]
    pub share_policy: bool,
}

#[cfg(feature = "torch")]
#[pymethods]
impl PyMappoConfig {
    #[new]
    #[pyo3(signature = (num_agents, obs_dim, global_state_dim, gamma=0.99, gae_lambda=0.95))]
    fn new(
        num_agents: usize,
        obs_dim: i64,
        global_state_dim: i64,
        gamma: f64,
        gae_lambda: f64,
    ) -> Self {
        Self {
            num_agents,
            obs_dim,
            global_state_dim,
            hidden_dim: 128,
            learning_rate: 1e-3,
            gamma,
            gae_lambda,
            clip_coef: 0.2,
            update_epochs: 4,
            share_policy: true,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PyMappoConfig(num_agents={}, obs_dim={})",
            self.num_agents, self.obs_dim
        )
    }
}

/// Create the PufferLib Python module
#[pymodule]
fn pufferlib_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PufferPythonEnv>()?;
    #[cfg(feature = "torch")]
    m.add_class::<PyMappoConfig>()?;
    Ok(())
}
