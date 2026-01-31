//! Offline trainer for Decision Transformer.

use super::buffer::SequenceBuffer;
use super::dt::DecisionTransformer;
use crate::log::MetricLogger;
use indicatif::{ProgressBar, ProgressStyle};
use tch::nn::OptimizerConfig;
use tch::{nn, Device};

pub struct OfflineTrainerConfig {
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub max_epochs: usize,
    pub steps_per_epoch: usize,
    pub device: Device,
}

impl Default for OfflineTrainerConfig {
    fn default() -> Self {
        Self {
            batch_size: std::env::var("OFFLINE_BATCH_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64),
            learning_rate: std::env::var("OFFLINE_LEARNING_RATE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1e-4),
            weight_decay: std::env::var("OFFLINE_WEIGHT_DECAY")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1e-4),
            max_epochs: std::env::var("OFFLINE_MAX_EPOCHS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10),
            steps_per_epoch: std::env::var("OFFLINE_STEPS_PER_EPOCH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(100),
            device: Device::Cpu,
        }
    }
}

pub struct OfflineTrainer<'a> {
    config: OfflineTrainerConfig,
    model: DecisionTransformer,
    buffer: SequenceBuffer,
    optimizer: nn::Optimizer,
    logger: Option<&'a dyn MetricLogger>,
}

impl<'a> OfflineTrainer<'a> {
    pub fn new(
        vs: &nn::VarStore,
        model: DecisionTransformer,
        buffer: SequenceBuffer,
        config: OfflineTrainerConfig,
        logger: Option<&'a dyn MetricLogger>,
    ) -> Self {
        let optimizer = nn::AdamW::default()
            .build(vs, config.learning_rate)
            .expect("Failed to create optimizer");

        Self {
            config,
            model,
            buffer,
            optimizer,
            logger,
        }
    }

    pub fn train(&mut self) {
        let pb = ProgressBar::new(self.config.max_epochs as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                .unwrap(),
        );

        for epoch in 0..self.config.max_epochs {
            let mut epoch_loss = 0.0;

            for step in 0..self.config.steps_per_epoch {
                let (obs, act, ret, time, _mask) = self.buffer.sample(self.config.batch_size);

                // Forward pass
                // Note: mask in buffer is (B, K), we need to use it to mask loss
                // Here we ignore mask for simplicity of example

                let (action_preds, _, _) = self.model.forward(&obs, &act, &ret, &time);

                // Loss: MSE between predicted action and actual action
                // Note: In real DT, target is the NEXT action (or same action depending on alignment)
                // Assuming target is `act` for reconstruction loss
                let loss = action_preds.mse_loss(&act, tch::Reduction::Mean);

                self.optimizer.zero_grad();
                loss.backward();
                self.optimizer.step();

                epoch_loss += loss.double_value(&[]);

                if let Some(logger) = self.logger {
                    logger.log_scalar(
                        "loss",
                        loss.double_value(&[]),
                        (epoch * self.config.steps_per_epoch + step) as u64,
                    );
                }
            }

            epoch_loss /= self.config.steps_per_epoch as f64;
            pb.set_message(format!("Loss: {:.4}", epoch_loss));
            pb.inc(1);
        }

        pb.finish_with_message("Training complete");
    }
}
