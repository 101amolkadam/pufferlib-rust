# Backend Abstraction Guide

*Technical specification for adding new ML backends (Burn, Luminal, ONNX).*

---

## Overview

PufferLib supports multiple ML backends through a trait-based abstraction. This allows users to choose the best backend for their deployment scenario:

| Backend | Use Case | Pros | Cons |
|:--------|:---------|:-----|:-----|
| **LibTorch** | Research, GPU training | Full PyTorch compatibility | Large binary, complex setup |
| **Candle** | Deployment, edge | Pure Rust, small binary | Limited ops |
| **Burn** | Cross-platform | Multi-backend, WGPU | Newer ecosystem |
| **ONNX** | Inference only | Universal format | No training |

---

## Backend Trait

### File: `crates/pufferlib/src/backend/mod.rs`

```rust
//! Backend abstraction for ML operations.

mod torch;
mod candle;
mod burn;
mod onnx;

use std::path::Path;

/// Core tensor trait
pub trait Tensor: Clone + Send + Sync {
    type Device;
    
    /// Create zeros tensor
    fn zeros(shape: &[i64], device: &Self::Device) -> Self;
    
    /// Create tensor from slice
    fn from_slice(data: &[f32], shape: &[i64], device: &Self::Device) -> Self;
    
    /// Get shape
    fn shape(&self) -> Vec<i64>;
    
    /// Reshape
    fn reshape(&self, shape: &[i64]) -> Self;
    
    /// Matrix multiply
    fn matmul(&self, other: &Self) -> Self;
    
    /// Element-wise add
    fn add(&self, other: &Self) -> Self;
    
    /// ReLU activation
    fn relu(&self) -> Self;
    
    /// Softmax
    fn softmax(&self, dim: i64) -> Self;
    
    /// Mean
    fn mean(&self) -> f64;
    
    /// Detach from computation graph
    fn detach(&self) -> Self;
    
    /// Convert to f32 vec
    fn to_vec(&self) -> Vec<f32>;
}

/// ML Backend trait
pub trait Backend: Send + Sync {
    type Tensor: Tensor;
    type Device: Clone + Send + Sync;
    type VarStore;
    type Optimizer;
    
    /// Get default device (CPU or GPU)
    fn default_device() -> Self::Device;
    
    /// Create a variable store for model parameters
    fn create_var_store(device: &Self::Device) -> Self::VarStore;
    
    /// Create optimizer
    fn create_optimizer(
        vs: &Self::VarStore,
        lr: f64,
    ) -> Self::Optimizer;
    
    /// Zero gradients
    fn zero_grad(optimizer: &Self::Optimizer);
    
    /// Backward pass
    fn backward(loss: &Self::Tensor);
    
    /// Optimizer step
    fn optimizer_step(optimizer: &mut Self::Optimizer);
    
    /// Save model to file
    fn save(vs: &Self::VarStore, path: impl AsRef<Path>) -> Result<(), BackendError>;
    
    /// Load model from file
    fn load(vs: &mut Self::VarStore, path: impl AsRef<Path>) -> Result<(), BackendError>;
    
    /// Export to ONNX (if supported)
    fn export_onnx(
        vs: &Self::VarStore,
        input_shape: &[i64],
        path: impl AsRef<Path>,
    ) -> Result<(), BackendError>;
}

#[derive(Debug)]
pub enum BackendError {
    IoError(std::io::Error),
    SerializationError(String),
    UnsupportedOperation(String),
}
```

---

## LibTorch Backend

### File: `crates/pufferlib/src/backend/torch.rs`

```rust
//! LibTorch (tch-rs) backend implementation.

use super::{Backend, BackendError, Tensor as TensorTrait};
use tch::{nn, Tensor, Device, Kind};
use std::path::Path;

pub struct TorchBackend;

impl TensorTrait for Tensor {
    type Device = Device;
    
    fn zeros(shape: &[i64], device: &Self::Device) -> Self {
        Tensor::zeros(shape, (Kind::Float, *device))
    }
    
    fn from_slice(data: &[f32], shape: &[i64], device: &Self::Device) -> Self {
        Tensor::from_slice(data)
            .reshape(shape)
            .to_device(*device)
    }
    
    fn shape(&self) -> Vec<i64> {
        self.size()
    }
    
    fn reshape(&self, shape: &[i64]) -> Self {
        self.reshape(shape)
    }
    
    fn matmul(&self, other: &Self) -> Self {
        self.matmul(other)
    }
    
    fn add(&self, other: &Self) -> Self {
        self + other
    }
    
    fn relu(&self) -> Self {
        self.relu()
    }
    
    fn softmax(&self, dim: i64) -> Self {
        self.softmax(dim, Kind::Float)
    }
    
    fn mean(&self) -> f64 {
        self.mean(Kind::Float).double_value(&[])
    }
    
    fn detach(&self) -> Self {
        self.detach()
    }
    
    fn to_vec(&self) -> Vec<f32> {
        Vec::<f32>::try_from(self.flatten(0, -1)).unwrap()
    }
}

impl Backend for TorchBackend {
    type Tensor = Tensor;
    type Device = Device;
    type VarStore = nn::VarStore;
    type Optimizer = nn::Optimizer;
    
    fn default_device() -> Self::Device {
        if tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        }
    }
    
    fn create_var_store(device: &Self::Device) -> Self::VarStore {
        nn::VarStore::new(*device)
    }
    
    fn create_optimizer(vs: &Self::VarStore, lr: f64) -> Self::Optimizer {
        nn::Adam::default().build(vs, lr).unwrap()
    }
    
    fn zero_grad(optimizer: &Self::Optimizer) {
        optimizer.zero_grad();
    }
    
    fn backward(loss: &Self::Tensor) {
        loss.backward();
    }
    
    fn optimizer_step(optimizer: &mut Self::Optimizer) {
        optimizer.step();
    }
    
    fn save(vs: &Self::VarStore, path: impl AsRef<Path>) -> Result<(), BackendError> {
        vs.save(path).map_err(|e| BackendError::IoError(
            std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
        ))
    }
    
    fn load(vs: &mut Self::VarStore, path: impl AsRef<Path>) -> Result<(), BackendError> {
        vs.load(path).map_err(|e| BackendError::IoError(
            std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
        ))
    }
    
    fn export_onnx(
        _vs: &Self::VarStore,
        _input_shape: &[i64],
        _path: impl AsRef<Path>,
    ) -> Result<(), BackendError> {
        // tch-rs ONNX export requires tracing
        Err(BackendError::UnsupportedOperation(
            "Use torch.jit.trace in Python for ONNX export".into()
        ))
    }
}
```

---

## Burn Backend

### File: `crates/pufferlib/src/backend/burn.rs`

```rust
//! Burn framework backend implementation.

use super::{Backend, BackendError, Tensor as TensorTrait};
use burn::prelude::*;
use burn::backend::Wgpu;
use std::path::Path;

pub type BurnDevice = burn::backend::wgpu::WgpuDevice;
pub type BurnTensor = Tensor<Wgpu, 2>;  // 2D tensor example

pub struct BurnBackend;

impl TensorTrait for BurnTensor {
    type Device = BurnDevice;
    
    fn zeros(shape: &[i64], device: &Self::Device) -> Self {
        let shape: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        Tensor::zeros(shape, device)
    }
    
    fn from_slice(data: &[f32], shape: &[i64], device: &Self::Device) -> Self {
        let shape: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        Tensor::from_floats(data, device).reshape(shape)
    }
    
    fn shape(&self) -> Vec<i64> {
        self.dims().iter().map(|&x| x as i64).collect()
    }
    
    fn reshape(&self, shape: &[i64]) -> Self {
        let shape: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        self.clone().reshape(shape)
    }
    
    fn matmul(&self, other: &Self) -> Self {
        self.clone().matmul(other.clone())
    }
    
    fn add(&self, other: &Self) -> Self {
        self.clone().add(other.clone())
    }
    
    fn relu(&self) -> Self {
        burn::tensor::activation::relu(self.clone())
    }
    
    fn softmax(&self, dim: i64) -> Self {
        burn::tensor::activation::softmax(self.clone(), dim as usize)
    }
    
    fn mean(&self) -> f64 {
        self.clone().mean().into_scalar() as f64
    }
    
    fn detach(&self) -> Self {
        self.clone().detach()
    }
    
    fn to_vec(&self) -> Vec<f32> {
        self.clone().into_data().convert().value
    }
}

impl Backend for BurnBackend {
    type Tensor = BurnTensor;
    type Device = BurnDevice;
    type VarStore = (); // Burn uses Record system
    type Optimizer = (); // Simplified
    
    fn default_device() -> Self::Device {
        BurnDevice::default()
    }
    
    fn create_var_store(_device: &Self::Device) -> Self::VarStore {
        ()
    }
    
    fn create_optimizer(_vs: &Self::VarStore, _lr: f64) -> Self::Optimizer {
        ()
    }
    
    fn zero_grad(_optimizer: &Self::Optimizer) {}
    
    fn backward(_loss: &Self::Tensor) {
        // Burn uses GradientsParams pattern
    }
    
    fn optimizer_step(_optimizer: &mut Self::Optimizer) {}
    
    fn save(_vs: &Self::VarStore, _path: impl AsRef<Path>) -> Result<(), BackendError> {
        // Use Burn's Record system
        Ok(())
    }
    
    fn load(_vs: &mut Self::VarStore, _path: impl AsRef<Path>) -> Result<(), BackendError> {
        Ok(())
    }
    
    fn export_onnx(
        _vs: &Self::VarStore,
        _input_shape: &[i64],
        path: impl AsRef<Path>,
    ) -> Result<(), BackendError> {
        // Burn supports ONNX export
        // burn::record::OnnxFileRecorder::new()
        Ok(())
    }
}
```

---

## Generic Policy

```rust
//! Generic policy that works across backends.

use crate::backend::{Backend, Tensor as TensorTrait};

/// Backend-agnostic policy trait
pub trait GenericPolicy<B: Backend>: Send {
    /// Forward pass
    fn forward(
        &self,
        observations: &B::Tensor,
        state: Option<Vec<B::Tensor>>,
    ) -> (B::Tensor, B::Tensor, Option<Vec<B::Tensor>>);
    
    /// Get initial recurrent state
    fn initial_state(&self, batch_size: i64) -> Option<Vec<B::Tensor>>;
}

/// Generic MLP that works with any backend
pub struct GenericMlp<B: Backend> {
    weights: Vec<B::Tensor>,
    biases: Vec<B::Tensor>,
    device: B::Device,
}

impl<B: Backend> GenericMlp<B> {
    pub fn new(
        layer_sizes: &[i64],
        device: B::Device,
    ) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        
        for window in layer_sizes.windows(2) {
            let (in_dim, out_dim) = (window[0], window[1]);
            // Initialize with random weights
            weights.push(B::Tensor::zeros(&[out_dim, in_dim], &device));
            biases.push(B::Tensor::zeros(&[out_dim], &device));
        }
        
        Self { weights, biases, device }
    }
    
    pub fn forward(&self, x: &B::Tensor) -> B::Tensor {
        let mut out = x.clone();
        
        for (i, (w, b)) in self.weights.iter().zip(&self.biases).enumerate() {
            out = out.matmul(w).add(b);
            
            // ReLU for all but last layer
            if i < self.weights.len() - 1 {
                out = out.relu();
            }
        }
        
        out
    }
}

impl<B: Backend> GenericPolicy<B> for GenericMlp<B> {
    fn forward(
        &self,
        observations: &B::Tensor,
        _state: Option<Vec<B::Tensor>>,
    ) -> (B::Tensor, B::Tensor, Option<Vec<B::Tensor>>) {
        let logits = self.forward(observations);
        let value = B::Tensor::zeros(&[observations.shape()[0], 1], &self.device);
        (logits, value, None)
    }
    
    fn initial_state(&self, _batch_size: i64) -> Option<Vec<B::Tensor>> {
        None
    }
}
```

---

## ONNX Export

### File: `crates/pufferlib/src/backend/onnx.rs`

```rust
//! ONNX export and inference.

use std::path::Path;

/// ONNX model wrapper for inference
pub struct OnnxModel {
    session: ort::Session,
}

impl OnnxModel {
    /// Load ONNX model
    pub fn load(path: impl AsRef<Path>) -> Result<Self, OnnxError> {
        let session = ort::Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_model_from_file(path)?;
        
        Ok(Self { session })
    }
    
    /// Run inference
    pub fn forward(&self, input: &[f32], shape: &[i64]) -> Result<Vec<f32>, OnnxError> {
        let input_tensor = ort::Value::from_array(
            ort::MemoryInfo::new_cpu(ort::AllocatorType::Arena, ort::MemType::Default)?,
            shape.iter().map(|&x| x as usize).collect::<Vec<_>>(),
            input,
        )?;
        
        let outputs = self.session.run(vec![input_tensor])?;
        
        let output_tensor = outputs[0].extract_tensor::<f32>()?;
        Ok(output_tensor.view().as_slice().unwrap().to_vec())
    }
}

#[derive(Debug)]
pub struct OnnxError(String);

impl From<ort::Error> for OnnxError {
    fn from(e: ort::Error) -> Self {
        OnnxError(e.to_string())
    }
}
```

---

## Adding a New Backend

### Checklist

1. **Create module**: `src/backend/newbackend.rs`
2. **Implement Tensor trait**: All tensor operations
3. **Implement Backend trait**: VarStore, Optimizer, save/load
4. **Add feature flag**: `Cargo.toml`
5. **Adapt policies**: Create NewBackendMlp, etc.
6. **Test**: Ensure parity with existing backends
7. **Document**: Add to this guide

### Example: Adding Luminal

```rust
// src/backend/luminal.rs
use luminal::prelude::*;

pub struct LuminalBackend;

impl Backend for LuminalBackend {
    // Implement all trait methods
    // Luminal uses a graph-based execution model
}
```

---

## Dependencies

```toml
[dependencies]
# LibTorch (default)
tch = { version = "0.15", optional = true }

# Candle
candle-core = { version = "0.4", optional = true }
candle-nn = { version = "0.4", optional = true }

# Burn
burn = { version = "0.14", optional = true, default-features = false }
burn-wgpu = { version = "0.14", optional = true }

# ONNX Runtime
ort = { version = "2.0", optional = true }

[features]
default = ["torch"]
torch = ["tch"]
candle = ["candle-core", "candle-nn"]
burn = ["dep:burn", "burn-wgpu"]
onnx = ["ort"]
```

---

*Last updated: 2026-01-28*
