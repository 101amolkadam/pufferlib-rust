//! Distributed training utilities using local threads and channels.
use crossbeam_channel::{bounded, Receiver, Sender};
use std::sync::{Arc, Barrier};
use tch::{Kind, Tensor};

/// Distributed training configuration
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DistributedConfig {
    /// Total number of workers (threads/GPUs)
    pub world_size: usize,
    /// This worker's rank (0-indexed)
    pub rank: usize,
    /// Master node address (unused in local mode)
    pub master_addr: String,
    /// Master node port (unused in local mode)
    pub master_port: u16,
    /// Backend: "local" or "nccl"
    pub backend: String,
    /// Whether to use gradient accumulation
    pub gradient_accumulation_steps: usize,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            rank: 0,
            master_addr: "localhost".to_string(),
            master_port: 29500,
            backend: "local".to_string(),
            gradient_accumulation_steps: 1,
        }
    }
}

/// Trait for distributed communication
pub trait DistributedBackend: Send + Sync {
    /// Initialize the distributed environment
    fn init(&mut self) -> Result<(), DistributedError>;

    /// AllReduce operation (sum and average gradients)
    fn all_reduce(&self, tensor: &mut Tensor);

    /// Broadcast from rank 0
    fn broadcast(&self, tensor: &mut Tensor);

    /// Barrier synchronization
    fn barrier(&self);

    /// Get world size
    fn world_size(&self) -> usize;

    /// Get current rank
    fn rank(&self) -> usize;

    /// Check if this is the master process
    fn is_master(&self) -> bool {
        self.rank() == 0
    }

    /// Cleanup
    fn finalize(&mut self);
}

#[derive(Debug, thiserror::Error)]
pub enum DistributedError {
    #[error("Initialization failed: {0}")]
    InitFailed(String),
    #[error("Communication error: {0}")]
    CommunicationError(String),
}

/// A synchronization group for thread-local distributed training
pub struct SyncGroup {
    pub barrier: Arc<Barrier>,
    // For AllReduce: Each rank has a sender to the master, and a receiver from the master
    pub reduc_senders: Vec<Sender<Tensor>>,
    pub reduc_receivers: Vec<Receiver<Tensor>>,
    // For Broadcast: Master has senders to all, workers have receivers
    pub bc_senders: Vec<Sender<Tensor>>,
    pub bc_receivers: Vec<Receiver<Tensor>>,
}

impl SyncGroup {
    pub fn new(world_size: usize) -> Self {
        let mut reduc_senders = Vec::with_capacity(world_size);
        let mut reduc_receivers = Vec::with_capacity(world_size);
        let mut bc_senders = Vec::with_capacity(world_size);
        let mut bc_receivers = Vec::with_capacity(world_size);

        for _ in 0..world_size {
            let (rs, rr) = bounded(1);
            let (bs, br) = bounded(1);
            reduc_senders.push(rs);
            reduc_receivers.push(rr);
            bc_senders.push(bs);
            bc_receivers.push(br);
        }

        Self {
            barrier: Arc::new(Barrier::new(world_size)),
            reduc_senders,
            reduc_receivers,
            bc_senders,
            bc_receivers,
        }
    }
}

/// Core backend implementing Thread-Local Distributed training
pub struct ThreadDistributedBackend {
    pub config: DistributedConfig,
    pub group: Arc<SyncGroup>,
}

impl ThreadDistributedBackend {
    pub fn new(config: DistributedConfig, group: Arc<SyncGroup>) -> Self {
        Self { config, group }
    }
}

impl DistributedBackend for ThreadDistributedBackend {
    fn init(&mut self) -> Result<(), DistributedError> {
        self.barrier();
        Ok(())
    }

    fn all_reduce(&self, tensor: &mut Tensor) {
        if self.config.world_size <= 1 {
            return;
        }

        let rank = self.config.rank;
        let ws = self.config.world_size;

        if rank == 0 {
            // Master: Collect from all
            let mut sum = tensor.shallow_clone();
            for i in 1..ws {
                let grad = self.group.reduc_receivers[i].recv().unwrap();
                sum = sum + grad.to_device(tensor.device());
            }
            let avg = sum / (ws as f64);

            // Broadcast back
            for i in 1..ws {
                self.group.bc_senders[i].send(avg.shallow_clone()).unwrap();
            }
            *tensor = avg;
        } else {
            // Worker: Send to master
            self.group.reduc_senders[rank]
                .send(tensor.shallow_clone())
                .unwrap();
            // Receive average
            *tensor = self.group.bc_receivers[rank]
                .recv()
                .unwrap()
                .to_device(tensor.device());
        }
    }

    fn broadcast(&self, tensor: &mut Tensor) {
        if self.config.world_size <= 1 {
            return;
        }

        let rank = self.config.rank;
        let ws = self.config.world_size;

        if rank == 0 {
            for i in 1..ws {
                self.group.bc_senders[i]
                    .send(tensor.shallow_clone())
                    .unwrap();
            }
        } else {
            *tensor = self.group.bc_receivers[rank]
                .recv()
                .unwrap()
                .to_device(tensor.device());
        }
    }

    fn barrier(&self) {
        self.group.barrier.wait();
    }

    fn world_size(&self) -> usize {
        self.config.world_size
    }

    fn rank(&self) -> usize {
        self.config.rank
    }

    fn finalize(&mut self) {
        self.barrier();
    }
}

use crate::policy::{HasVarStore, Policy};
use crate::training::{TrainMetrics, Trainer};
use crate::vector::{ObservationBatch, VecEnvBackend};

/// Metrics for distributed training
#[derive(Debug)]
pub struct DistributedMetrics {
    pub local_metrics: TrainMetrics,
    pub world_size: usize,
}

/// A trainer wrapper that handles distributed synchronization
pub struct DistributedTrainer<P: Policy + HasVarStore, V: VecEnvBackend, B: DistributedBackend> {
    pub trainer: Trainer<P, V>,
    pub backend: B,
    pub config: DistributedConfig,
}

impl<P: Policy + HasVarStore, V: VecEnvBackend, B: DistributedBackend> DistributedTrainer<P, V, B> {
    pub fn new(trainer: Trainer<P, V>, backend: B, config: DistributedConfig) -> Self {
        Self {
            trainer,
            backend,
            config,
        }
    }

    /// Synchronize model weights across all workers (rank 0 is source)
    pub fn sync_weights(&mut self) {
        let vars: Vec<Tensor> = self
            .trainer
            .policy
            .var_store()
            .variables()
            .values()
            .map(|t| t.shallow_clone())
            .collect();
        for mut tensor in vars {
            self.backend.broadcast(&mut tensor);
        }
        self.backend.barrier();
    }

    /// Perform a distributed training step
    pub fn train_step(&mut self, obs: &ObservationBatch) -> Option<DistributedMetrics> {
        // 1. Collect rollouts locally
        self.trainer.collect_rollout(obs);

        // 2. Perform local update
        // Future: Split this into compute_gradients -> all_reduce -> apply_gradients
        let metrics = self.trainer.update();

        // 3. Barrier to keep workers in sync
        self.backend.barrier();

        if self.backend.is_master() {
            Some(DistributedMetrics {
                local_metrics: metrics,
                world_size: self.config.world_size,
            })
        } else {
            None
        }
    }
}
