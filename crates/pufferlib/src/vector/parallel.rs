//! Parallel vectorization backend.
//!
//! Runs environments in parallel using rayon for high throughput.

use super::vecenv::{ObservationBatch, VecEnvBackend, VecEnvResult};
use crate::env::{EnvInfo, PufferEnv};
use crate::spaces::DynSpace;
use ndarray::{Array2, ArrayD, IxDyn};
use rayon::prelude::*;

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// Parallel vectorization backend using rayon
pub struct Parallel<E: PufferEnv> {
    /// Environments owned by the backend
    envs: Vec<E>,
    /// Number of environments
    num_envs: usize,
    /// Cached observation shape
    obs_shape: Vec<usize>,
    /// Cached observation and action spaces
    obs_space: DynSpace,
    action_space: DynSpace,
    /// Shared memory buffer for observations
    obs_buffer: Option<Box<dyn super::SharedBuffer>>,
}

impl<E: PufferEnv> Parallel<E> {
    /// Create a new parallel backend
    pub fn new<F>(env_creator: F, num_envs: usize) -> Self
    where
        F: Fn() -> E + Send + Sync,
    {
        assert!(num_envs > 0, "Number of environments must be > 0");

        // Create first env to get spaces
        let first_env = env_creator();
        let obs_space = first_env.observation_space();
        let action_space = first_env.action_space();
        let obs_shape = obs_space.shape().to_vec();

        // Create all envs in parallel
        let mut envs: Vec<_> = (0..num_envs)
            .into_par_iter()
            .map(|_| env_creator())
            .collect();

        // Replace first env to ensure we reuse the one created for space detection
        envs[0] = first_env;

        // Initialize shared buffer if on Windows mapping
        #[cfg(target_os = "windows")]
        let obs_buffer = {
            let total_obs_size = num_envs * obs_shape.iter().product::<usize>();
            let name = format!("pufferlib_obs_{}", rand::random::<u32>());
            match super::Win32SharedBuffer::new(&name, total_obs_size) {
                Ok(buf) => Some(Box::new(buf) as Box<dyn super::SharedBuffer>),
                Err(e) => {
                    tracing::warn!("Failed to create shared buffer: {}. Falling back to heap.", e);
                    None
                }
            }
        };
        #[cfg(not(target_os = "windows"))]
        let obs_buffer = None;

        Self {
            envs,
            num_envs,
            obs_shape,
            obs_space,
            action_space,
            obs_buffer,
        }
    }
}

impl<E: PufferEnv> VecEnvBackend for Parallel<E> {
    fn observation_space(&self) -> DynSpace {
        self.obs_space.clone()
    }

    fn action_space(&self) -> DynSpace {
        self.action_space.clone()
    }

    fn num_envs(&self) -> usize {
        self.num_envs
    }

    fn reset(&mut self, seed: Option<u64>) -> (ObservationBatch, Vec<EnvInfo>) {
        let obs_size = self.obs_shape.iter().product::<usize>();
        let buffer_ptr = self.obs_buffer.as_ref().map(|b| SendPtr(b.as_ptr()));

        let results: Vec<_> = self
            .envs
            .par_iter_mut()
            .enumerate()
            .map(|(i, env)| {
                let env_seed = seed.map(|s| s + i as u64);
                let (obs, info) = env.reset(env_seed);
                
                // Zero-copy write if buffer exists
                if let Some(send_ptr) = buffer_ptr {
                    let ptr = send_ptr.0;
                    unsafe {
                        let offset_ptr = ptr.add(i * obs_size);
                        std::ptr::copy_nonoverlapping(obs.as_ptr(), offset_ptr, obs_size);
                    }
                }
                
                (obs, info)
            })
            .collect();

        let infos: Vec<_> = results.iter().map(|(_, i)| i.clone()).collect();

        // Compatibility: Return an owned Array2 by stacking.
        // The data is ALSO in the shared buffer for the trainer to use (Zero-Copy Read).
        let observations: Vec<_> = results.into_iter().map(|(o, _)| o).collect();
        let flat_obs: Vec<f32> = observations.iter().flat_map(|o| o.iter().copied()).collect();
        let obs_array = Array2::from_shape_vec((self.num_envs, obs_size), flat_obs).expect("Failed to create observation array");

        (ObservationBatch::Cpu(obs_array), infos)
    }

    fn step(&mut self, actions: &Array2<f32>) -> VecEnvResult {
        let obs_size = self.obs_shape.iter().product::<usize>();
        let buffer_ptr = self.obs_buffer.as_ref().map(|b| SendPtr(b.as_ptr()));

        let results: Vec<_> = self
            .envs
            .par_iter_mut()
            .enumerate()
            .map(|(i, env)| {
                let action_row = actions.row(i);
                let action =
                    ArrayD::from_shape_vec(IxDyn(&[action_row.len()]), action_row.to_vec())
                        .expect("Failed to create action array");

                let result = if env.is_done() {
                    let (obs, info) = env.reset(None);
                    (obs, 0.0, false, false, info)
                } else {
                    let res = env.step(&action);
                    (
                        res.observation,
                        res.reward,
                        res.terminated,
                        res.truncated,
                        res.info,
                    )
                };

                // Zero-copy write
                if let Some(send_ptr) = buffer_ptr {
                    let ptr = send_ptr.0;
                    unsafe {
                        let offset_ptr = ptr.add(i * obs_size);
                        std::ptr::copy_nonoverlapping(result.0.as_ptr(), offset_ptr, obs_size);
                    }
                }
                
                result
            })
            .collect();

        let rewards: Vec<_> = results.iter().map(|(_, r, _, _, _)| *r).collect();
        let terminated: Vec<_> = results.iter().map(|(_, _, t, _, _)| *t).collect();
        let truncated: Vec<_> = results.iter().map(|(_, _, _, t, _)| *t).collect();
        let infos: Vec<_> = results.iter().map(|(_, _, _, _, i)| i.clone()).collect();

        let observations: Vec<_> = results.into_iter().map(|(o, _, _, _, _)| o).collect();
        let flat_obs: Vec<f32> = observations.iter().flat_map(|o| o.iter().copied()).collect();
        let obs_array = Array2::from_shape_vec((self.num_envs, obs_size), flat_obs).expect("Failed to create observation array");

        VecEnvResult {
            observations: ObservationBatch::Cpu(obs_array),
            rewards,
            terminated,
            truncated,
            infos,
        }
    }

    fn close(&mut self) {
        self.envs.par_iter_mut().for_each(|env| {
            env.close();
        });
    }
}
