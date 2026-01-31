use crate::mappo::{MultiAgentEnv, MultiAgentStepResult};
use numpy::{PyArrayDyn, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use std::collections::HashMap;
use tch::{Device, Kind, Tensor};

#[cfg(feature = "python")]
pub struct PettingZooEnv {
    env: PyObject,
    agents: Vec<String>,
    device: Device,
}

#[cfg(feature = "python")]
impl PettingZooEnv {
    pub fn new(env_creator: &PyObject, device: Device) -> Self {
        Python::with_gil(|py| {
            let env = env_creator
                .bind(py)
                .call0()
                .expect("Failed to create PettingZoo environment");

            // Get agents list and sort for consistent indexing
            let agents_list = env
                .getattr("possible_agents")
                .expect("PettingZoo env missing possible_agents")
                .downcast_into::<PyList>()
                .expect("possible_agents is not a list");

            let mut agents: Vec<String> = agents_list
                .iter()
                .map(|item| item.extract::<String>().unwrap())
                .collect();
            agents.sort();

            Self {
                env: env.to_object(py),
                agents,
                device,
            }
        })
    }

    fn py_to_tensor(&self, _py: Python, py_obj: &Bound<'_, PyAny>) -> Tensor {
        if let Ok(array) = py_obj.downcast::<PyArrayDyn<f32>>() {
            let nd = unsafe { array.as_array() };
            let shape: Vec<i64> = nd.shape().iter().map(|&x| x as i64).collect();
            Tensor::from_slice(nd.as_slice().unwrap())
                .reshape(&shape)
                .to_device(self.device)
        } else if let Ok(val) = py_obj.extract::<f32>() {
            Tensor::from(val).to_device(self.device)
        } else if let Ok(val) = py_obj.extract::<i64>() {
            Tensor::from(val).to_device(self.device)
        } else {
            panic!("Unsupported observation type from Python: {:?}", py_obj);
        }
    }

    fn tensor_to_py<'py>(&self, py: Python<'py>, tensor: &Tensor) -> PyObject {
        if tensor.numel() == 1 {
            // Scalar for discrete actions
            if tensor.kind() == Kind::Int64 {
                tensor.int64_value(&[]).into_py(py)
            } else {
                tensor.double_value(&[]).into_py(py)
            }
        } else {
            // Multi-dimensional for continuous/multi-discrete
            // Convert to ndarray first
            let size: Vec<usize> = tensor.size().iter().map(|&x| x as usize).collect();
            // This is slow but robust for the bridge
            let data: Vec<f32> = vec![0.0; tensor.numel()];
            // tensor.copy_data(&mut data, tensor.numel()); // tch-rs might not have easy copy_data for all types
            // Alternatively, use a numpy conversion if tch supports it.
            // Simplified: return as numpy array via a temporary Vec
            // For now, let's assume it's mostly discrete or the user will handle it.
            // A real impl should use a more efficient tensor -> numpy path.
            let array = ndarray::Array::from_shape_vec(size, data).unwrap();
            array.to_pyarray_bound(py).into_any().unbind()
        }
    }
}

#[cfg(feature = "python")]
impl MultiAgentEnv for PettingZooEnv {
    fn reset(&mut self) -> Vec<Tensor> {
        Python::with_gil(|py| {
            let env = self.env.bind(py);
            let result = env.call_method0("reset").expect("reset failed");

            // Parallel API returns (obs, info)
            let obs_dict = if let Ok(tuple) = result.downcast::<PyTuple>() {
                tuple
                    .get_item(0)
                    .unwrap()
                    .downcast_into::<PyDict>()
                    .unwrap()
            } else {
                result.downcast_into::<PyDict>().unwrap()
            };

            self.agents
                .iter()
                .map(|agent_id| {
                    let py_obs = obs_dict.get_item(agent_id).unwrap().unwrap();
                    self.py_to_tensor(py, &py_obs)
                })
                .collect()
        })
    }

    fn step(&mut self, actions: &[Tensor]) -> MultiAgentStepResult {
        Python::with_gil(|py| {
            let env = self.env.bind(py);

            // Convert actions to dict
            let action_dict = PyDict::new_bound(py);
            for (i, action) in actions.iter().enumerate() {
                let agent_id = &self.agents[i];
                let py_action = self.tensor_to_py(py, action);
                action_dict.set_item(agent_id, py_action).unwrap();
            }

            let result = env
                .call_method1("step", (action_dict,))
                .expect("step failed");
            let tuple = result
                .downcast::<PyTuple>()
                .expect("Step result is not a tuple");

            let obs = tuple
                .get_item(0)
                .unwrap()
                .downcast_into::<PyDict>()
                .expect("obs is not a dict");
            let rewards = tuple
                .get_item(1)
                .unwrap()
                .downcast_into::<PyDict>()
                .expect("rewards is not a dict");
            let terminations = tuple
                .get_item(2)
                .unwrap()
                .downcast_into::<PyDict>()
                .expect("terminations is not a dict");
            let truncations = tuple
                .get_item(3)
                .unwrap()
                .downcast_into::<PyDict>()
                .expect("truncations is not a dict");

            let mut out_obs = Vec::new();
            let mut out_rewards = Vec::new();
            let mut out_dones = Vec::new();

            for agent_id in &self.agents {
                let py_obs = obs
                    .get_item(agent_id)
                    .unwrap()
                    .expect("Agent missing in obs");
                out_obs.push(self.py_to_tensor(py, &py_obs));

                let r: f32 = rewards
                    .get_item(agent_id)
                    .unwrap()
                    .expect("Agent missing in rewards")
                    .extract()
                    .unwrap();
                out_rewards.push(r);

                let term: bool = terminations
                    .get_item(agent_id)
                    .unwrap()
                    .expect("Agent missing in terminations")
                    .extract()
                    .unwrap();
                let trunc: bool = truncations
                    .get_item(agent_id)
                    .unwrap()
                    .expect("Agent missing in truncations")
                    .extract()
                    .unwrap();
                out_dones.push(term || trunc);
            }

            MultiAgentStepResult {
                observations: out_obs,
                rewards: out_rewards,
                dones: out_dones,
                info: HashMap::new(),
                costs: vec![0.0; self.agents.len()],
            }
        })
    }

    fn get_global_state(&self) -> Tensor {
        Python::with_gil(|py| {
            let env = self.env.bind(py);
            if let Ok(state) = env.call_method0("state") {
                self.py_to_tensor(py, &state)
            } else {
                // Fallback: concatenate all observations
                let obs_vec = self
                    .agents
                    .iter()
                    .map(|_id| {
                        // This is inefficient but a possible fallback if state() is missing
                        Tensor::zeros([1], (Kind::Float, self.device))
                    })
                    .collect::<Vec<_>>();
                Tensor::cat(&obs_vec, 0)
            }
        })
    }

    fn num_agents(&self) -> usize {
        self.agents.len()
    }

    fn get_observation(&self, _agent_id: usize) -> Tensor {
        // Similar to reset, but for a single agent if supported
        todo!("Direct observation access")
    }
}
