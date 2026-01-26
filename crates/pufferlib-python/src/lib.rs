//! PyO3 bindings for PufferLib.

use ndarray::ArrayD;
use numpy::{PyArrayDyn, PyArrayMethods, ToPyArray};
use pufferlib::env::{PufferEnv, StepResult};
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
        let py_obs = obs.to_pyarray_bound(py).to_owned().into();
        let py_info = PyDict::new_bound(py).into();
        Ok((py_obs, py_info))
    }

    /// Step the environment
    #[allow(clippy::type_complexity)]
    fn step(&mut self, py: Python, action: Py<PyArrayDyn<f32>>) -> PyResult<StepResultTuple> {
        // Convert Python array to ndarray
        let action_bound = action.bind(py);
        let action_array: ArrayD<f32> = unsafe { action_bound.as_array().to_owned() };

        let result: StepResult = self.env.step(&action_array);

        let py_obs = result.observation.to_pyarray_bound(py).to_owned().into();
        let py_info = PyDict::new_bound(py).into();

        Ok((
            py_obs,
            result.reward,
            result.terminated,
            result.truncated,
            py_info,
        ))
    }
}

/// Create the PufferLib Python module
#[pymodule]
fn pufferlib_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PufferPythonEnv>()?;
    Ok(())
}
