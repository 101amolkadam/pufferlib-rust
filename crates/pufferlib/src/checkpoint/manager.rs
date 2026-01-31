//! Checkpoint manager for automatic rotation and best model tracking.

use super::state::Checkpointable;
#[cfg(feature = "std")]
/// Checkpoint manager for automatic rotation and best model tracking.
use crate::{PufferError, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// Configuration for checkpoint management.
#[derive(Clone, Debug)]
pub struct CheckpointConfig {
    /// Directory to store checkpoints
    pub checkpoint_dir: PathBuf,
    /// Save checkpoint every N epochs
    pub save_every: u64,
    /// Keep only the last N checkpoints (0 = keep all)
    pub keep_last: usize,
    /// Also save a "best" checkpoint based on reward
    pub save_best: bool,
    /// Whether to save experience buffer state (can be large)
    pub save_buffer: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: PathBuf::from("checkpoints"),
            save_every: 10,
            keep_last: 5,
            save_best: true,
            save_buffer: false,
        }
    }
}

impl CheckpointConfig {
    /// Create a new config with the given directory.
    pub fn new(checkpoint_dir: impl Into<PathBuf>) -> Self {
        Self {
            checkpoint_dir: checkpoint_dir.into(),
            ..Default::default()
        }
    }

    /// Set save frequency.
    pub fn save_every(mut self, epochs: u64) -> Self {
        self.save_every = epochs;
        self
    }

    /// Set number of checkpoints to keep.
    pub fn keep_last(mut self, n: usize) -> Self {
        self.keep_last = n;
        self
    }

    /// Enable/disable best checkpoint tracking.
    pub fn save_best(mut self, enabled: bool) -> Self {
        self.save_best = enabled;
        self
    }

    /// Enable/disable buffer state saving.
    pub fn save_buffer(mut self, enabled: bool) -> Self {
        self.save_buffer = enabled;
        self
    }
}

/// Manages checkpoint lifecycle.
///
/// Handles saving, loading, rotation, and best checkpoint tracking.
///
/// # Example
///
/// ```ignore
/// let config = CheckpointConfig::new("./checkpoints")
///     .save_every(100)
///     .keep_last(3)
///     .save_best(true);
///
/// let mut manager = CheckpointManager::new(config);
///
/// // In training loop:
/// if let Some(path) = manager.maybe_save(&trainer, epoch, reward)? {
///     println!("Saved checkpoint: {}", path.display());
/// }
///
/// // To resume:
/// if let Some(epoch) = manager.load_latest(&mut trainer)? {
///     println!("Resumed from epoch {}", epoch);
/// }
/// ```
pub struct CheckpointManager {
    config: CheckpointConfig,
    best_reward: f64,
}

impl CheckpointManager {
    /// Create a new checkpoint manager.
    pub fn new(config: CheckpointConfig) -> Self {
        // Create checkpoint directory
        if let Err(e) = fs::create_dir_all(&config.checkpoint_dir) {
            tracing::warn!("Failed to create checkpoint directory: {}", e);
        }

        Self {
            config,
            best_reward: f64::NEG_INFINITY,
        }
    }

    /// Get the checkpoint directory path.
    pub fn checkpoint_dir(&self) -> &Path {
        &self.config.checkpoint_dir
    }

    /// Save a checkpoint if conditions are met (epoch divisible by save_every).
    ///
    /// Returns the path to the saved checkpoint, or None if no save was performed.
    pub fn maybe_save<T: Checkpointable>(
        &mut self,
        trainable: &T,
        epoch: u64,
        reward: f64,
    ) -> Result<Option<PathBuf>> {
        // Check if we should save this epoch
        if epoch == 0 || epoch % self.config.save_every != 0 {
            return Ok(None);
        }

        self.save(trainable, epoch, reward)
    }

    /// Force save a checkpoint regardless of epoch.
    pub fn save<T: Checkpointable>(
        &mut self,
        trainable: &T,
        epoch: u64,
        reward: f64,
    ) -> Result<Option<PathBuf>> {
        let data = trainable.save_state()?;

        let filename = format!("checkpoint_epoch_{:06}.bin", epoch);
        let path = self.config.checkpoint_dir.join(&filename);

        // Write checkpoint
        fs::write(&path, &data)?;
        tracing::info!(path = %path.display(), epoch, "Saved checkpoint");

        // Save best checkpoint
        if self.config.save_best && reward > self.best_reward {
            self.best_reward = reward;
            let best_path = self.config.checkpoint_dir.join("checkpoint_best.bin");
            fs::copy(&path, &best_path)?;
            tracing::info!(reward, "New best checkpoint!");
        }

        // Cleanup old checkpoints
        if self.config.keep_last > 0 {
            self.cleanup_old_checkpoints()?;
        }

        Ok(Some(path))
    }

    /// Load the latest checkpoint.
    ///
    /// Returns the epoch number if a checkpoint was loaded, or None if no checkpoints exist.
    pub fn load_latest<T: Checkpointable>(&self, trainable: &mut T) -> Result<Option<u64>> {
        let latest = self.find_latest_checkpoint()?;

        if let Some(path) = latest {
            let data = fs::read(&path)?;
            trainable.load_state(&data)?;

            // Extract epoch from filename
            let epoch = self.extract_epoch_from_path(&path);
            tracing::info!(path = %path.display(), epoch, "Loaded checkpoint");

            Ok(Some(epoch))
        } else {
            Ok(None)
        }
    }

    /// Load the best checkpoint.
    ///
    /// Returns true if the best checkpoint was loaded, false if it doesn't exist.
    pub fn load_best<T: Checkpointable>(&self, trainable: &mut T) -> Result<bool> {
        let best_path = self.config.checkpoint_dir.join("checkpoint_best.bin");

        if best_path.exists() {
            let data = fs::read(&best_path)?;
            trainable.load_state(&data)?;
            tracing::info!("Loaded best checkpoint");
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Load a specific checkpoint file.
    pub fn load_from_path<T: Checkpointable>(
        &self,
        trainable: &mut T,
        path: impl AsRef<Path>,
    ) -> Result<()> {
        let data = fs::read(path.as_ref())?;
        trainable.load_state(&data)?;
        tracing::info!(path = %path.as_ref().display(), "Loaded checkpoint");
        Ok(())
    }

    /// Find the latest checkpoint file.
    fn find_latest_checkpoint(&self) -> Result<Option<PathBuf>> {
        let entries = match fs::read_dir(&self.config.checkpoint_dir) {
            Ok(e) => e,
            Err(_) => return Ok(None),
        };

        let mut checkpoints: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("checkpoint_epoch_") && n.ends_with(".bin"))
                    .unwrap_or(false)
            })
            .collect();

        checkpoints.sort();
        Ok(checkpoints.pop())
    }

    /// List all checkpoint files in order.
    pub fn list_checkpoints(&self) -> Result<Vec<PathBuf>> {
        let entries = match fs::read_dir(&self.config.checkpoint_dir) {
            Ok(e) => e,
            Err(_) => return Ok(Vec::new()),
        };

        let mut checkpoints: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("checkpoint_epoch_") && n.ends_with(".bin"))
                    .unwrap_or(false)
            })
            .collect();

        checkpoints.sort();
        Ok(checkpoints)
    }

    /// Remove old checkpoints, keeping only the last N.
    fn cleanup_old_checkpoints(&self) -> Result<()> {
        let mut checkpoints = self.list_checkpoints()?;

        while checkpoints.len() > self.config.keep_last {
            let old = checkpoints.remove(0);
            if let Err(e) = fs::remove_file(&old) {
                tracing::warn!(path = %old.display(), "Failed to remove old checkpoint: {}", e);
            } else {
                tracing::debug!(path = %old.display(), "Removed old checkpoint");
            }
        }

        Ok(())
    }

    /// Extract epoch number from checkpoint filename.
    fn extract_epoch_from_path(&self, path: &Path) -> u64 {
        path.file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_prefix("checkpoint_epoch_"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    /// Mock checkpointable for testing
    struct MockTrainable {
        data: Vec<u8>,
    }

    impl Checkpointable for MockTrainable {
        fn save_state(&self) -> Result<Vec<u8>> {
            Ok(self.data.clone())
        }

        fn load_state(&mut self, data: &[u8]) -> Result<()> {
            self.data = data.to_vec();
            Ok(())
        }
    }

    #[test]
    fn test_checkpoint_config_builder() {
        let config = CheckpointConfig::new("./test")
            .save_every(50)
            .keep_last(10)
            .save_best(false)
            .save_buffer(true);

        assert_eq!(config.checkpoint_dir, PathBuf::from("./test"));
        assert_eq!(config.save_every, 50);
        assert_eq!(config.keep_last, 10);
        assert!(!config.save_best);
        assert!(config.save_buffer);
    }

    #[test]
    fn test_maybe_save_respects_frequency() {
        let dir = tempdir().unwrap();
        let config = CheckpointConfig::new(dir.path()).save_every(5);
        let mut manager = CheckpointManager::new(config);
        let trainable = MockTrainable {
            data: vec![1, 2, 3],
        };

        // Epoch 0 should not save
        assert!(manager.maybe_save(&trainable, 0, 0.0).unwrap().is_none());

        // Epoch 3 should not save
        assert!(manager.maybe_save(&trainable, 3, 0.0).unwrap().is_none());

        // Epoch 5 should save
        assert!(manager.maybe_save(&trainable, 5, 0.0).unwrap().is_some());

        // Epoch 10 should save
        assert!(manager.maybe_save(&trainable, 10, 0.0).unwrap().is_some());
    }

    #[test]
    fn test_save_and_load() {
        let dir = tempdir().unwrap();
        let config = CheckpointConfig::new(dir.path());
        let mut manager = CheckpointManager::new(config);

        let trainable = MockTrainable {
            data: vec![1, 2, 3, 4, 5],
        };

        // Save
        let path = manager.save(&trainable, 10, 100.0).unwrap();
        assert!(path.is_some());

        // Load into new trainable
        let mut loaded = MockTrainable { data: vec![] };
        let epoch = manager.load_latest(&mut loaded).unwrap();

        assert_eq!(epoch, Some(10));
        assert_eq!(loaded.data, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_best_checkpoint_tracking() {
        let dir = tempdir().unwrap();
        let config = CheckpointConfig::new(dir.path()).save_best(true);
        let mut manager = CheckpointManager::new(config);

        let trainable = MockTrainable {
            data: vec![1, 2, 3],
        };

        // Save with reward 50
        manager.save(&trainable, 1, 50.0).unwrap();

        // Save with reward 100 (new best)
        manager.save(&trainable, 2, 100.0).unwrap();

        // Save with reward 75 (not new best)
        manager.save(&trainable, 3, 75.0).unwrap();

        // Best should still be epoch 2's data
        assert!(dir.path().join("checkpoint_best.bin").exists());
    }

    #[test]
    fn test_cleanup_old_checkpoints() {
        let dir = tempdir().unwrap();
        let config = CheckpointConfig::new(dir.path()).save_every(1).keep_last(2);
        let mut manager = CheckpointManager::new(config);

        let trainable = MockTrainable { data: vec![1] };

        // Save 5 checkpoints
        for epoch in 1..=5 {
            manager.save(&trainable, epoch, 0.0).unwrap();
        }

        // Should only have 2 left
        let checkpoints = manager.list_checkpoints().unwrap();
        assert_eq!(checkpoints.len(), 2);

        // Should be epochs 4 and 5
        assert!(checkpoints[0]
            .to_string_lossy()
            .contains("checkpoint_epoch_000004"));
        assert!(checkpoints[1]
            .to_string_lossy()
            .contains("checkpoint_epoch_000005"));
    }

    #[test]
    fn test_load_from_path() {
        let dir = tempdir().unwrap();
        let config = CheckpointConfig::new(dir.path());
        let manager = CheckpointManager::new(config);

        // Create a checkpoint file manually
        let path = dir.path().join("custom_checkpoint.bin");
        let mut file = fs::File::create(&path).unwrap();
        file.write_all(&[9, 8, 7]).unwrap();

        // Load it
        let mut trainable = MockTrainable { data: vec![] };
        manager.load_from_path(&mut trainable, &path).unwrap();

        assert_eq!(trainable.data, vec![9, 8, 7]);
    }
}
