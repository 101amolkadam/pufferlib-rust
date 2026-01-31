# Python Interoperability Guide

*Technical specification for PyO3 bindings, HuggingFace Hub, and Python ecosystem integration.*

---

## Overview

PufferLib provides seamless Python interoperability through:
- **PyO3 Bindings**: Use Rust envs/policies from Python
- **HuggingFace Hub**: Model sharing and versioning
- **Gymnasium Bridge**: Use Python envs from Rust

---

## PyO3 Bindings

### File: `crates/pufferlib-python/src/lib.rs`

```rust
//! Python bindings for PufferLib.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArray2, IntoPyArray};

/// Python-exposed PufferEnv wrapper
#[pyclass]
pub struct PyPufferEnv {
    inner: Box<dyn pufferlib::env::PufferEnv>,
}

#[pymethods]
impl PyPufferEnv {
    /// Create a new environment by name
    #[staticmethod]
    fn from_name(py: Python, name: &str) -> PyResult<Self> {
        let env = match name {
            "CartPole" => Box::new(pufferlib_envs::CartPole::new()),
            "Bandit" => Box::new(pufferlib_envs::Bandit::new(10)),
            _ => return Err(PyValueError::new_err(format!("Unknown env: {}", name))),
        };
        Ok(Self { inner: env })
    }
    
    /// Reset the environment
    fn reset(&mut self, py: Python, seed: Option<u64>) -> PyResult<(Py<PyArray1<f32>>, PyObject)> {
        let (obs, info) = self.inner.reset(seed);
        let obs_array = obs.as_slice().unwrap().to_vec().into_pyarray(py).to_owned();
        let info_dict = PyDict::new(py);
        Ok((obs_array, info_dict.into()))
    }
    
    /// Step the environment
    fn step(&mut self, py: Python, action: &PyArray1<f32>) -> PyResult<(
        Py<PyArray1<f32>>,  // observation
        f32,                 // reward
        bool,                // terminated
        bool,                // truncated
        PyObject,            // info
    )> {
        let action_slice = unsafe { action.as_slice()? };
        let action_array = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[action_slice.len()]),
            action_slice.to_vec(),
        ).map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        let result = self.inner.step(&action_array);
        
        let obs_array = result.observation.as_slice().unwrap().to_vec().into_pyarray(py).to_owned();
        let info_dict = PyDict::new(py);
        
        Ok((
            obs_array,
            result.reward,
            result.terminated,
            result.truncated,
            info_dict.into(),
        ))
    }
    
    /// Get observation space info
    #[getter]
    fn observation_space(&self, py: Python) -> PyResult<PyObject> {
        let info = self.inner.env_info();
        let dict = PyDict::new(py);
        dict.set_item("shape", info.obs_shape.clone())?;
        dict.set_item("dtype", "float32")?;
        Ok(dict.into())
    }
    
    /// Get action space info
    #[getter]
    fn action_space(&self, py: Python) -> PyResult<PyObject> {
        let info = self.inner.env_info();
        let dict = PyDict::new(py);
        dict.set_item("shape", info.action_shape.clone())?;
        dict.set_item("dtype", "float32")?;
        Ok(dict.into())
    }
}

/// Python-exposed VecEnv
#[pyclass]
pub struct PyVecEnv {
    inner: pufferlib::vector::VecEnv<pufferlib_envs::CartPole, pufferlib::vector::ParallelBackend>,
}

#[pymethods]
impl PyVecEnv {
    #[new]
    fn new(num_envs: usize) -> PyResult<Self> {
        let config = pufferlib::vector::VecEnvConfig::default();
        let envs: Vec<_> = (0..num_envs)
            .map(|_| pufferlib_envs::CartPole::new())
            .collect();
        
        let vecenv = pufferlib::vector::VecEnv::new(envs, config)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        Ok(Self { inner: vecenv })
    }
    
    fn reset(&mut self, py: Python) -> PyResult<Py<PyArray2<f32>>> {
        let obs = self.inner.reset();
        let shape = obs.shape();
        let flat_obs: Vec<f32> = obs.iter().cloned().collect();
        
        PyArray2::from_vec(py, flat_obs, (shape[0], shape[1]))
            .map(|a| a.to_owned())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    
    fn step(&mut self, py: Python, actions: &PyArray2<f32>) -> PyResult<(
        Py<PyArray2<f32>>,  // observations
        Py<PyArray1<f32>>,  // rewards
        Py<PyArray1<bool>>, // dones
        PyObject,           // infos
    )> {
        // Convert actions and step
        // ...similar conversion logic...
        todo!()
    }
}

/// Module initialization
#[pymodule]
fn pufferlib_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyPufferEnv>()?;
    m.add_class::<PyVecEnv>()?;
    Ok(())
}
```

---

## Python Usage

```python
# Installation
# pip install pufferlib-rs

import pufferlib_rs
import numpy as np

# Create environment
env = pufferlib_rs.PyPufferEnv.from_name("CartPole")

# Gymnasium-compatible API
obs, info = env.reset(seed=42)
print(f"Observation shape: {obs.shape}")

# Step
action = np.array([0.0], dtype=np.float32)
obs, reward, terminated, truncated, info = env.step(action)
print(f"Reward: {reward}, Done: {terminated or truncated}")

# VecEnv for parallelism
vec_env = pufferlib_rs.PyVecEnv(num_envs=16)
obs = vec_env.reset()
print(f"Batched observations: {obs.shape}")  # (16, obs_dim)
```

---

## HuggingFace Hub Integration

### Model Upload

```rust
//! HuggingFace Hub integration.

use std::path::Path;

pub struct HubConfig {
    pub repo_id: String,
    pub token: Option<String>,
}

/// Upload model to HuggingFace Hub
pub fn push_to_hub(
    model_path: impl AsRef<Path>,
    config: &HubConfig,
    commit_message: &str,
) -> Result<String, HubError> {
    // Use huggingface_hub Python library via subprocess
    let script = format!(
        r#"
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="{}",
    path_in_repo="model.bin",
    repo_id="{}",
    commit_message="{}",
)
print("uploaded")
"#,
        model_path.as_ref().display(),
        config.repo_id,
        commit_message,
    );
    
    let output = std::process::Command::new("python")
        .args(["-c", &script])
        .env("HF_TOKEN", config.token.as_deref().unwrap_or(""))
        .output()?;
    
    if output.status.success() {
        Ok(format!("https://huggingface.co/{}", config.repo_id))
    } else {
        Err(HubError::UploadFailed(
            String::from_utf8_lossy(&output.stderr).to_string()
        ))
    }
}

/// Download model from HuggingFace Hub
pub fn pull_from_hub(
    repo_id: &str,
    filename: &str,
    local_dir: impl AsRef<Path>,
) -> Result<std::path::PathBuf, HubError> {
    let script = format!(
        r#"
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="{}", filename="{}", local_dir="{}")
print(path)
"#,
        repo_id,
        filename,
        local_dir.as_ref().display(),
    );
    
    let output = std::process::Command::new("python")
        .args(["-c", &script])
        .output()?;
    
    if output.status.success() {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(std::path::PathBuf::from(path))
    } else {
        Err(HubError::DownloadFailed(
            String::from_utf8_lossy(&output.stderr).to_string()
        ))
    }
}

#[derive(Debug)]
pub enum HubError {
    IoError(std::io::Error),
    UploadFailed(String),
    DownloadFailed(String),
}
```

### Usage

```rust
// Push trained model
push_to_hub(
    "checkpoints/best_model.bin",
    &HubConfig {
        repo_id: "username/pufferlib-cartpole-ppo".into(),
        token: std::env::var("HF_TOKEN").ok(),
    },
    "Upload trained PPO model",
)?;

// Pull model for inference
let model_path = pull_from_hub(
    "username/pufferlib-cartpole-ppo",
    "model.bin",
    "./models",
)?;
```

---

## Gymnasium Bridge (Python → Rust)

```rust
//! Use Python Gymnasium environments from Rust.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use ndarray::ArrayD;

/// Wrapper for Python Gymnasium environments
pub struct GymnasiumEnv {
    py_env: PyObject,
}

impl GymnasiumEnv {
    /// Create from environment ID
    pub fn make(env_id: &str) -> PyResult<Self> {
        Python::with_gil(|py| {
            let gym = PyModule::import(py, "gymnasium")?;
            let env = gym.call_method1("make", (env_id,))?;
            Ok(Self { py_env: env.into() })
        })
    }
}

impl pufferlib::env::PufferEnv for GymnasiumEnv {
    fn reset(&mut self, seed: Option<u64>) -> (ArrayD<f32>, std::collections::HashMap<String, String>) {
        Python::with_gil(|py| {
            let kwargs = pyo3::types::PyDict::new(py);
            if let Some(s) = seed {
                kwargs.set_item("seed", s).ok();
            }
            
            let result = self.py_env.call_method(py, "reset", (), Some(kwargs)).unwrap();
            let tuple: &PyTuple = result.extract(py).unwrap();
            let obs: Vec<f32> = tuple.get_item(0).unwrap().extract().unwrap();
            
            (ArrayD::from_shape_vec(ndarray::IxDyn(&[obs.len()]), obs).unwrap(), 
             std::collections::HashMap::new())
        })
    }
    
    fn step(&mut self, action: &ArrayD<f32>) -> pufferlib::env::StepResult {
        Python::with_gil(|py| {
            let action_list = action.as_slice().unwrap().to_vec();
            let result = self.py_env.call_method1(py, "step", (action_list,)).unwrap();
            let tuple: &PyTuple = result.extract(py).unwrap();
            
            let obs: Vec<f32> = tuple.get_item(0).unwrap().extract().unwrap();
            let reward: f32 = tuple.get_item(1).unwrap().extract().unwrap();
            let terminated: bool = tuple.get_item(2).unwrap().extract().unwrap();
            let truncated: bool = tuple.get_item(3).unwrap().extract().unwrap();
            
            pufferlib::env::StepResult {
                observation: ArrayD::from_shape_vec(ndarray::IxDyn(&[obs.len()]), obs).unwrap(),
                reward,
                terminated,
                truncated,
                info: std::collections::HashMap::new(),
            }
        })
    }
    
    fn env_info(&self) -> pufferlib::env::EnvInfo {
        // Extract from Python observation/action spaces
        pufferlib::env::EnvInfo {
            obs_shape: vec![4],
            action_shape: vec![1],
            num_agents: 1,
        }
    }
}
```

---

## SB3 Compatibility

```python
# Use PufferLib Rust envs with Stable Baselines3
import pufferlib_rs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Wrapper for SB3 compatibility
class SB3Wrapper:
    def __init__(self, env_name):
        self.env = pufferlib_rs.PyPufferEnv.from_name(env_name)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action):
        return self.env.step(action)
    
    @property
    def observation_space(self):
        import gymnasium as gym
        info = self.env.observation_space
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=tuple(info["shape"]))
    
    @property
    def action_space(self):
        import gymnasium as gym
        return gym.spaces.Discrete(2)  # For CartPole

# Train with SB3
env = SB3Wrapper("CartPole")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

---

## Dependencies

```toml
[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
numpy = "0.20"

[lib]
name = "pufferlib_rs"
crate-type = ["cdylib"]
```

### Build Instructions

```bash
# Install maturin
pip install maturin

# Build and install
cd crates/pufferlib-python
maturin develop --release

# Or build wheel
maturin build --release
pip install target/wheels/pufferlib_rs-*.whl
```

---

*Last updated: 2026-01-28*
