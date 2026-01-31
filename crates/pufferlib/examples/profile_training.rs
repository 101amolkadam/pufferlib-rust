use pufferlib::policy::MlpPolicy;
use pufferlib::prelude::*;
use pufferlib::training::{Trainer, TrainerConfig};
use pufferlib::vector::{AsyncVecEnv, Parallel};
use pufferlib_envs::CartPole;
use tch::Device;

fn main() -> anyhow::Result<()> {
    println!("Initializing profiling script...");

    let num_envs = 64;
    let device = if Device::cuda_is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    println!("Profiling with {} environments on {:?}", num_envs, device);

    // Create environments - use Parallel::new with a closure
    let backend = Parallel::new(|| CartPole::new(), num_envs);
    let async_backend = AsyncVecEnv::new(backend);
    let vecenv = VecEnv::new(async_backend, VecEnvConfig::default());

    let obs_space = vecenv.observation_space();
    let action_space = vecenv.action_space();

    // Create policy - MLP policy takes 3 arguments: &obs, &act, device
    let policy = MlpPolicy::new(&obs_space, &action_space, device);

    // Trainer configuration
    let config = TrainerConfig::default()
        .with_timesteps(1_000_000)
        .with_lr(3e-4);

    let mut config = config;
    config.device = device;
    config.use_amp = true;
    config.batch_size = 16384;
    config.num_minibatches = 4;
    config.update_epochs = 4;

    // Trainer::new takes 4 arguments: vecenv, policy, config, device
    let mut trainer = Trainer::new(vecenv, policy, config, device);

    println!("--- Starting Profiling Run (5 epochs) ---");
    for i in 0..5 {
        let (obs, _) = trainer.vecenv.reset(None);
        trainer.collect_rollout(&obs);
        let metrics = trainer.update();

        println!("Epoch {}: SPS: {:.2}", i, metrics.sps);
        println!(
            "  Rollout Time: {:.4}s (Env: {:.4}s)",
            metrics.rollout_time, metrics.env_time
        );
        println!("  Update Time:  {:.4}s", metrics.update_time);
        println!("  Total Reward: {:.2}", metrics.reward);
    }
    println!("--- Profiling Complete ---");

    Ok(())
}
