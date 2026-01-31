//! Sequence buffer for offline RL.

use std::collections::VecDeque;
use tch::{Device, Tensor};

/// A single trajectory of experience.
#[derive(Clone, Debug)]
pub struct Trajectory {
    pub observations: Vec<Vec<f32>>,
    pub actions: Vec<Vec<f32>>,
    pub rewards: Vec<f32>,
    pub returns_to_go: Vec<f32>,
    pub terminals: Vec<bool>,
}

impl Trajectory {
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
            actions: Vec::new(),
            rewards: Vec::new(),
            returns_to_go: Vec::new(),
            terminals: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.observations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    /// Compute returns-to-go for the trajectory.
    pub fn compute_returns_to_go(&mut self, gamma: f32) {
        let mut running_return = 0.0;
        self.returns_to_go = vec![0.0; self.len()];
        for i in (0..self.len()).rev() {
            running_return = self.rewards[i] + gamma * running_return;
            self.returns_to_go[i] = running_return;
        }
    }
}

/// Buffer for storing and sampling trajectories.
pub struct SequenceBuffer {
    trajectories: Vec<Trajectory>,
    context_len: usize,
    device: Device,
    /// Total number of timesteps stored
    total_timesteps: usize,
    /// Cumulative lengths for efficient sampling
    cumulative_lengths: Vec<usize>,
}

impl SequenceBuffer {
    pub fn new(context_len: usize, device: Device) -> Self {
        Self {
            trajectories: Vec::new(),
            context_len,
            device,
            total_timesteps: 0,
            cumulative_lengths: vec![0],
        }
    }

    /// Add a pre-computed trajectory.
    pub fn add_trajectory(&mut self, traj: Trajectory) {
        let len = traj.len();
        if len > 0 {
            self.total_timesteps += len;
            self.cumulative_lengths.push(self.total_timesteps);
            self.trajectories.push(traj);
        }
    }

    /// Sample a batch of subsequences.
    /// Returns (states, actions, returns, timesteps, masks)
    pub fn sample(&self, batch_size: usize) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        let mut batch_obs = Vec::with_capacity(batch_size);
        let mut batch_act = Vec::with_capacity(batch_size);
        let mut batch_ret = Vec::with_capacity(batch_size);
        let mut batch_time = Vec::with_capacity(batch_size);
        let mut batch_mask = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            // Pick rand trajectory weighted by length
            let idx = rand::random::<usize>() % self.total_timesteps;
            let traj_idx = match self.cumulative_lengths.binary_search_by(|&x| x.cmp(&idx)) {
                Ok(i) => i.min(self.trajectories.len() - 1),
                Err(i) => (i - 1).min(self.trajectories.len() - 1),
            };

            let traj = &self.trajectories[traj_idx];

            // Pick random start index
            let start_idx = rand::random::<usize>() % traj.len();

            // Extract subsequence
            let end_idx = (start_idx + self.context_len).min(traj.len());
            let real_len = end_idx - start_idx;

            // Pad needed?
            let pad_len = self.context_len - real_len;

            // Prepare padding
            let obs_dim = traj.observations[0].len();
            let act_dim = traj.actions[0].len();

            let mut obs_seq = vec![vec![0.0; obs_dim]; pad_len];
            let mut act_seq = vec![vec![0.0; act_dim]; pad_len];
            let mut ret_seq = vec![0.0; pad_len];
            let mut time_seq = vec![0; pad_len];
            let mut mask_seq = vec![0; pad_len];

            // Fill real data
            for i in 0..real_len {
                obs_seq[i] = traj.observations[start_idx + i].clone();
                act_seq[i] = traj.actions[start_idx + i].clone();
                ret_seq[i] = traj.returns_to_go[start_idx + i];
                time_seq[i] = (start_idx + i) as i64;
                mask_seq[i] = 1;
            }

            let obs_flat: Vec<f32> = obs_seq.into_iter().flatten().collect();
            let act_flat: Vec<f32> = act_seq.into_iter().flatten().collect();

            batch_obs.push(
                Tensor::from_slice(&obs_flat).reshape(&[self.context_len as i64, obs_dim as i64]),
            );
            batch_act.push(
                Tensor::from_slice(&act_flat).reshape(&[self.context_len as i64, act_dim as i64]),
            );
            batch_ret.push(Tensor::from_slice(&ret_seq));
            batch_time.push(Tensor::from_slice(&time_seq));
            batch_mask.push(Tensor::from_slice(&mask_seq));
        }

        let obs = Tensor::stack(&batch_obs, 0).to(self.device); // [B, K, O]
        let act = Tensor::stack(&batch_act, 0).to(self.device); // [B, K, A]
        let ret = Tensor::stack(&batch_ret, 0).to(self.device).unsqueeze(2); // [B, K, 1]
        let time = Tensor::stack(&batch_time, 0).to(self.device); // [B, K]
        let mask = Tensor::stack(&batch_mask, 0).to(self.device); // [B, K]

        (obs, act, ret, time, mask)
    }
}
