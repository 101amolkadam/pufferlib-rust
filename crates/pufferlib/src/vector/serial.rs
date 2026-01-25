//! Serial (sequential) vectorization backend.
//!
//! Runs environments one at a time in a single thread.
//! Useful for debugging and small-scale experiments.

use super::vecenv::{ObservationBatch, VecEnvBackend, VecEnvResult};
use crate::env::{EnvInfo, PufferEnv};
use crate::spaces::DynSpace;
use ndarray::{Array2, ArrayD, IxDyn};

/// Serial vectorization backend
pub struct Serial<E: PufferEnv> {
    /// Created environments
    envs: Vec<E>,
    /// Number of environments
    num_envs: usize,
    /// Cached observation shape
    obs_shape: Vec<usize>,
}

impl<E: PufferEnv> Serial<E> {
    /// Create a new serial backend
    pub fn new<F>(env_creator: F, num_envs: usize) -> Self
    where
        F: Fn() -> E,
    {
        // Create first env to get spaces
        let first_env = env_creator();
        let obs_space = first_env.observation_space();
        let obs_shape = obs_space.shape().to_vec();

        // Create all envs
        let mut envs = Vec::with_capacity(num_envs);
        envs.push(first_env);
        for _ in 1..num_envs {
            envs.push(env_creator());
        }

        Self {
            envs,
            num_envs,
            obs_shape,
        }
    }
}

impl<E: PufferEnv> VecEnvBackend for Serial<E> {
    fn observation_space(&self) -> DynSpace {
        self.envs[0].observation_space()
    }

    fn action_space(&self) -> DynSpace {
        self.envs[0].action_space()
    }

    fn num_envs(&self) -> usize {
        self.num_envs
    }

    fn reset(&mut self, seed: Option<u64>) -> (ObservationBatch, Vec<EnvInfo>) {
        let mut observations = Vec::with_capacity(self.num_envs);
        let mut infos = Vec::with_capacity(self.num_envs);

        for (i, env) in self.envs.iter_mut().enumerate() {
            let env_seed = seed.map(|s| s + i as u64);
            let (obs, info) = env.reset(env_seed);
            observations.push(obs);
            infos.push(info);
        }

        // Stack observations into 2D array
        let flat_obs: Vec<f32> = observations
            .iter()
            .flat_map(|o| o.iter().copied())
            .collect();
        let obs_array =
            Array2::from_shape_vec((self.num_envs, self.obs_shape.iter().product()), flat_obs)
                .expect("Failed to create observation array");

        (ObservationBatch::Cpu(obs_array), infos)
    }

    fn step(&mut self, actions: &Array2<f32>) -> VecEnvResult {
        let obs_size = self.obs_shape.iter().product::<usize>();
        let mut observations = Vec::with_capacity(self.num_envs * obs_size);
        let mut rewards = Vec::with_capacity(self.num_envs);
        let mut terminated = Vec::with_capacity(self.num_envs);
        let mut truncated = Vec::with_capacity(self.num_envs);
        let mut infos = Vec::with_capacity(self.num_envs);

        for (i, env) in self.envs.iter_mut().enumerate() {
            // Get action for this env
            let action_row = actions.row(i);
            let action = ArrayD::from_shape_vec(IxDyn(&[action_row.len()]), action_row.to_vec())
                .expect("Failed to create action array");

            // Check if env needs reset
            if env.is_done() {
                let (obs, info) = env.reset(None);
                observations.extend(obs.into_iter());
                rewards.push(0.0);
                terminated.push(false);
                truncated.push(false);
                infos.push(info);
            } else {
                let result = env.step(&action);
                observations.extend(result.observation.into_iter());
                rewards.push(result.reward);
                terminated.push(result.terminated);
                truncated.push(result.truncated);
                infos.push(result.info);
            }
        }

        // Stack observations
        let obs_array = Array2::from_shape_vec((self.num_envs, obs_size), observations).unwrap();

        VecEnvResult {
            observations: ObservationBatch::Cpu(obs_array),
            rewards,
            terminated,
            truncated,
            infos,
        }
    }

    fn close(&mut self) {
        for env in &mut self.envs {
            env.close();
        }
    }
}
