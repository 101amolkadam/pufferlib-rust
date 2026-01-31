use pufferlib::policy::{MlpConfig, MlpPolicy};
use pufferlib::prelude::*;
use pufferlib::training::{Trainer, TrainerConfig};
use pufferlib::vector::{Parallel, VecEnv};
use pufferlib_envs::CartPole;
use tch::Device;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing profiling script...");

    let num_envs = 64;
    let device = if tch::utils::has_cuda() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    println!("Profiling with {} environments on {:?}", num_envs, device);

    // Create environments - use Parallel::new with a closure
    let vecenv = VecEnv::from_backend(Parallel::new(|| CartPole::new(), num_envs));

    let obs_size = vecenv.observation_space().shape()[0] as i64;
    let num_actions = 2i64; // CartPole

    // Create policy
    let config = MlpConfig::default();
    let policy = MlpPolicy::new(obs_size, num_actions, false, config, device);

    // Trainer configuration
    let mut config = TrainerConfig::default();
    config.total_timesteps = 1_000_000;
    config.learning_rate = 3e-4;
    config.device = device;
    config.use_amp = false; // Disable AMP for CPU profiling if needed
    config.batch_size = 16384;
    config.num_minibatches = 4;
    config.update_epochs = 4;

    // Trainer::new
    let mut trainer = Trainer::<_, _, pufferlib::training::optimizer::TorchOptimizer>::new(
        vecenv, policy, config, device,
    );

    println!("--- Starting Profiling Run (5 epochs) ---");
    trainer.train()?; // Run full training or just steps

    println!("--- Profiling Complete ---");

    Ok(())
}
