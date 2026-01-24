//! Parallel vectorization backend.
//!
//! Runs environments in parallel using rayon for high throughput.

use super::vecenv::{VecEnvBackend, VecEnvResult};
use crate::env::{EnvInfo, PufferEnv};
use crate::spaces::DynSpace;
use ndarray::{Array2, ArrayD, IxDyn};
use rayon::prelude::*;

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

        Self {
            envs,
            num_envs,
            obs_shape,
            obs_space,
            action_space,
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

    fn reset(&mut self, seed: Option<u64>) -> (Array2<f32>, Vec<EnvInfo>) {
        let results: Vec<_> = self
            .envs
            .par_iter_mut()
            .enumerate()
            .map(|(i, env)| {
                let env_seed = seed.map(|s| s + i as u64);
                env.reset(env_seed)
            })
            .collect();

        let observations: Vec<_> = results.iter().map(|(o, _)| o.clone()).collect();
        let infos: Vec<_> = results.into_iter().map(|(_, i)| i).collect();

        // Stack observations
        let flat_obs: Vec<f32> = observations
            .iter()
            .flat_map(|o| o.iter().copied())
            .collect();
        let obs_array =
            Array2::from_shape_vec((self.num_envs, self.obs_shape.iter().product()), flat_obs)
                .expect("Failed to create observation array");

        (obs_array, infos)
    }

    fn step(&mut self, actions: &Array2<f32>) -> VecEnvResult {
        let results: Vec<_> = self
            .envs
            .par_iter_mut()
            .enumerate()
            .map(|(i, env)| {
                let action_row = actions.row(i);
                let action =
                    ArrayD::from_shape_vec(IxDyn(&[action_row.len()]), action_row.to_vec())
                        .expect("Failed to create action array");

                if env.is_done() {
                    let (obs, info) = env.reset(None);
                    (obs, 0.0, false, false, info)
                } else {
                    let result = env.step(&action);
                    (
                        result.observation,
                        result.reward,
                        result.terminated,
                        result.truncated,
                        result.info,
                    )
                }
            })
            .collect();

        let observations: Vec<_> = results.iter().map(|(o, _, _, _, _)| o.clone()).collect();
        let rewards: Vec<_> = results.iter().map(|(_, r, _, _, _)| *r).collect();
        let terminated: Vec<_> = results.iter().map(|(_, _, t, _, _)| *t).collect();
        let truncated: Vec<_> = results.iter().map(|(_, _, _, t, _)| *t).collect();
        let infos: Vec<_> = results.into_iter().map(|(_, _, _, _, i)| i).collect();

        // Stack observations
        let flat_obs: Vec<f32> = observations
            .iter()
            .flat_map(|o| o.iter().copied())
            .collect();
        let obs_array =
            Array2::from_shape_vec((self.num_envs, self.obs_shape.iter().product()), flat_obs)
                .expect("Failed to create observation array");

        VecEnvResult {
            observations: obs_array,
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
