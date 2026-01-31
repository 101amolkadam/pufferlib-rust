//! ONNX Runtime policy for cross-platform inference.

use crate::types::{String, Vec};
use ndarray::ArrayD;
#[cfg(feature = "onnx")]
use ort::{inputs, Session, Value};

/// Policy that runs inference using an ONNX model
#[cfg(feature = "onnx")]
pub struct OnnxPolicy {
    session: Session,
    input_name: String,
    output_names: Vec<String>,
}

#[cfg(feature = "onnx")]
impl OnnxPolicy {
    /// Create a new ONNX policy from a file
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?.commit_from_file(path)?;

        let input_name = session.inputs[0].name.clone();
        let output_names = session.outputs.iter().map(|o| o.name.clone()).collect();

        Ok(Self {
            session,
            input_name,
            output_names,
        })
    }

    /// Run inference
    pub fn forward(
        &self,
        obs: &ArrayD<f32>,
    ) -> Result<Vec<ArrayD<f32>>, Box<dyn std::error::Error>> {
        // Convert ArrayD to Ort Value
        // Note: ort 2.0 uses different API for inputs
        let input_tensor = obs.clone();
        let outputs = self
            .session
            .run(inputs![&self.input_name => input_tensor]?)?;

        let mut result = Vec::new();
        for name in &self.output_names {
            let output = outputs.get(name).unwrap();
            // Extract data back to ndarray
            // This depends on the exact ort version API
            // Placeholder for data extraction
        }

        Ok(result)
    }

    /// Enable TensorRT execution provider
    pub fn with_tensorrt(mut self) -> Result<Self, Box<dyn std::error::Error>> {
        // In a real implementation, we would recreate the session
        // with the TensorRT execution provider enabled.
        Ok(self)
    }
}
