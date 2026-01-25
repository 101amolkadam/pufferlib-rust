//! PufferLib CLI
//!
//! Command-line interface for training and evaluating RL agents.

use anyhow::Result;
use clap::{Parser, Subcommand};
use ndarray::{ArrayD, IxDyn};
use tracing_subscriber::EnvFilter;

use pufferlib::env::PufferEnv;
use pufferlib_envs::{Bandit, CartPole, Memory, Squared};

// #[cfg(feature = "torch")]
// use pufferlib::vector::VecEnvBackend;

mod hpo;

#[derive(Parser)]
#[command(name = "puffer")]
#[command(version, about = "PufferLib - High-performance RL in Rust", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train an agent (requires --features torch)
    Train {
        /// Environment name
        #[arg(default_value = "cartpole")]
        env: String,

        /// Total timesteps
        #[arg(long, default_value = "100000")]
        timesteps: u64,

        /// Learning rate
        #[arg(long, default_value = "0.0003")]
        lr: f64,

        /// Number of environments
        #[arg(long, default_value = "4")]
        num_envs: usize,

        /// Policy type (mlp, lstm)
        #[arg(long, default_value = "mlp")]
        policy: String,
    },

    /// Evaluate an agent (random policy without --features torch)
    Eval {
        /// Environment name
        env: String,

        /// Number of episodes
        #[arg(long, default_value = "10")]
        episodes: usize,
    },

    /// List available environments
    List,

    /// Demo: Run environment interactively
    Demo {
        /// Environment name
        #[arg(default_value = "cartpole")]
        env: String,

        /// Number of steps
        #[arg(long, default_value = "100")]
        steps: usize,
    },
    
    /// Auto-Tune: Search for best hyperparameters (Protein)
    AutoTune {
        /// Environment name
        #[arg(default_value = "cartpole")]
        env: String,
        
        /// Number of trials
        #[arg(long, default_value = "20")]
        trials: usize,
        
        /// Steps per trial
        #[arg(long, default_value = "50000")]
        steps: u64,
    },
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            env: _env,
            timesteps: _timesteps,
            lr: _lr,
            num_envs: _num_envs,
            policy: _policy,
        } => {
            #[cfg(feature = "torch")]
            {
                train(&_env, _timesteps, _lr, _num_envs, &_policy)?;
            }
            #[cfg(not(feature = "torch"))]
            {
                tracing::error!("Training requires the 'torch' feature. Rebuild with:");
                tracing::error!("  cargo build --features torch");
                tracing::error!("Note: libtorch must be installed. See README.md for details.");
            }
        }
        Commands::Eval { env, episodes } => {
            eval(&env, episodes)?;
        }
        Commands::List => {
            list_envs();
        }
        Commands::Demo { env, steps } => {
            demo(&env, steps)?;
        }
        Commands::AutoTune { env, trials, steps } => {
            #[cfg(feature = "torch")]
            {
                autotune(&env, trials, steps)?;
            }
            #[cfg(not(feature = "torch"))]
            {
                tracing::error!("Auto-Tune requires the 'torch' feature.");
            }
        }
    }

    Ok(())
}

#[cfg(feature = "torch")]
#[allow(unused_imports)]
fn train(
    env_name: &str,
    timesteps: u64,
    lr: f64,
    num_envs: usize,
    policy_type: &str,
) -> Result<()> {
    use pufferlib::policy::{LstmPolicy, MlpConfig, MlpPolicy};
    use pufferlib::spaces::Space;
    use pufferlib::vector::{Parallel, Serial, VecEnv};

    tracing::info!(
        env = env_name,
        timesteps,
        lr,
        num_envs,
        policy = policy_type,
        "Starting training"
    );

    let device = if tch::Cuda::is_available() {
        tracing::info!("Using CUDA");
        tch::Device::Cuda(0)
    } else {
        tracing::info!("Using CPU");
        tch::Device::Cpu
    };

    // Helper for setup
    let trainer_config = pufferlib::training::TrainerConfig {
        total_timesteps: timesteps,
        learning_rate: lr,
        ..Default::default()
    };

    match env_name {
        "bandit" => {
            let make_env = || Bandit::new(4);
            let num_actions = 4i64;

            if num_envs > 1 {
                let envs = VecEnv::from_backend(Parallel::new(make_env, num_envs));
                let obs_size = envs.observation_space().shape()[0];
                if policy_type == "lstm" {
                    let policy = LstmPolicy::new(obs_size as i64, num_actions, 64, device);
                    tracing::info!(params = policy.num_parameters(), "Created LSTM policy");
                    run_training_thread(envs, policy, trainer_config, device)?;
                } else {
                    let config = MlpConfig::default();
                    let policy =
                        MlpPolicy::new(obs_size as i64, num_actions, false, config, device);
                    tracing::info!(params = policy.num_parameters(), "Created MLP policy");
                    run_training_thread(envs, policy, trainer_config, device)?;
                }
            } else {
                let envs = VecEnv::from_backend(Serial::new(make_env, num_envs));
                let obs_size = envs.observation_space().shape()[0];
                if policy_type == "lstm" {
                    let policy = LstmPolicy::new(obs_size as i64, num_actions, 64, device);
                    tracing::info!(params = policy.num_parameters(), "Created LSTM policy");
                    run_training_thread(envs, policy, trainer_config, device)?;
                } else {
                    let config = MlpConfig::default();
                    let policy =
                        MlpPolicy::new(obs_size as i64, num_actions, false, config, device);
                    tracing::info!(params = policy.num_parameters(), "Created MLP policy");
                    run_training_thread(envs, policy, trainer_config, device)?;
                }
            }
        }
        "cartpole" => {
            let make_env = || CartPole::new();
            let num_actions = 2i64;

            if num_envs > 1 {
                let envs = VecEnv::from_backend(Parallel::new(make_env, num_envs));
                let obs_size = envs.observation_space().shape()[0];
                if policy_type == "lstm" {
                    let policy = LstmPolicy::new(obs_size as i64, num_actions, 64, device);
                    tracing::info!(params = policy.num_parameters(), "Created LSTM policy");
                    run_training_thread(envs, policy, trainer_config, device)?;
                } else {
                    let config = MlpConfig::default();
                    let policy =
                        MlpPolicy::new(obs_size as i64, num_actions, false, config, device);
                    tracing::info!(params = policy.num_parameters(), "Created MLP policy");
                    run_training_thread(envs, policy, trainer_config, device)?;
                }
            } else {
                let envs = VecEnv::from_backend(Serial::new(make_env, num_envs));
                let obs_size = envs.observation_space().shape()[0];
                if policy_type == "lstm" {
                    let policy = LstmPolicy::new(obs_size as i64, num_actions, 64, device);
                    tracing::info!(params = policy.num_parameters(), "Created LSTM policy");
                    run_training_thread(envs, policy, trainer_config, device)?;
                } else {
                    let config = MlpConfig::default();
                    let policy =
                        MlpPolicy::new(obs_size as i64, num_actions, false, config, device);
                    tracing::info!(params = policy.num_parameters(), "Created MLP policy");
                    run_training_thread(envs, policy, trainer_config, device)?;
                }
            }
        }
        _ => {
            tracing::warn!(
                "Environment '{}' not fully implemented for training yet",
                env_name
            );
        }
    }

    Ok(())
}

#[cfg(feature = "torch")]
fn run_training_thread<B, P>(
    envs: pufferlib::vector::VecEnv<B>,
    policy: P,
    config: pufferlib::training::TrainerConfig,
    device: tch::Device,
) -> Result<()>
where
    B: pufferlib::vector::VecEnvBackend + 'static,
    P: pufferlib::policy::Policy + pufferlib::policy::HasVarStore + 'static,
{
    let mut trainer = pufferlib::training::Trainer::new(envs, policy, config, device);

    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024) // 8MB
        .spawn(move || {
            trainer.train().expect("Training failed");
        })?
        .join()
        .unwrap();

    Ok(())
}

fn eval(env_name: &str, episodes: usize) -> Result<()> {
    tracing::info!(
        env = env_name,
        episodes,
        "Starting evaluation (random policy)"
    );

    match env_name {
        "bandit" => {
            let mut env = Bandit::new(4);
            let mut total_score = 0.0;

            for ep in 0..episodes {
                let (_obs, _) = env.reset(Some(ep as u64));

                // Random action
                let action = ArrayD::from_elem(IxDyn(&[1]), (ep % 4) as f32);
                let result = env.step(&action);

                if let Some(score) = result.info.get("score") {
                    total_score += score;
                }
            }

            let avg_score = total_score / episodes as f32;
            tracing::info!(avg_score, "Evaluation complete");
        }
        "cartpole" => {
            let mut env = CartPole::new();
            let mut total_length = 0;

            for _ in 0..episodes {
                env.reset(None);
                let mut length = 0;

                while !env.is_done() {
                    let action = ArrayD::from_elem(IxDyn(&[1]), (length % 2) as f32);
                    env.step(&action);
                    length += 1;
                    if length > 500 {
                        break;
                    }
                }

                total_length += length;
            }

            let avg_length = total_length / episodes;
            tracing::info!(avg_length, "Average episode length with alternating policy");
        }
        "squared" => {
            let mut env = Squared::new(2);
            env.reset(Some(42));

            for _ in 0..episodes {
                let action = ArrayD::from_elem(IxDyn(&[1]), 0.0);
                let result = env.step(&action);
                if result.done() {
                    env.reset(None);
                }
            }
            tracing::info!("Squared evaluation complete");
        }
        "memory" => {
            let mut env = Memory::new(3, 0);
            env.reset(Some(42));

            for _ in 0..episodes {
                let action = ArrayD::from_elem(IxDyn(&[1]), 0.0);
                let result = env.step(&action);
                if result.done() {
                    env.reset(None);
                }
            }
            tracing::info!("Memory evaluation complete");
        }
        _ => {
            tracing::error!("Unknown environment: {}", env_name);
        }
    }

    Ok(())
}

fn demo(env_name: &str, steps: usize) -> Result<()> {
    tracing::info!(env = env_name, steps, "Running demo");

    match env_name {
        "cartpole" => {
            let mut env = CartPole::new();
            let (_, _) = env.reset(Some(42));

            for step in 0..steps {
                let action = ArrayD::from_elem(IxDyn(&[1]), (step % 2) as f32);
                let result = env.step(&action);

                if step % 10 == 0 {
                    if let Some(render) = env.render() {
                        println!("Step {}: {}", step, render);
                    }
                }

                if result.done() {
                    tracing::info!(step, "Episode ended, resetting");
                    env.reset(None);
                }
            }
        }
        "bandit" => {
            let mut env = Bandit::new(4);

            for step in 0..steps {
                let (_, _) = env.reset(None);
                let action = ArrayD::from_elem(IxDyn(&[1]), (step % 4) as f32);
                let result = env.step(&action);
                println!(
                    "Step {}: action={}, reward={}",
                    step,
                    step % 4,
                    result.reward
                );
            }
        }
        "memory" => {
            let mut env = Memory::new(3, 0);
            env.reset(Some(42));

            for step in 0..steps {
                let action = ArrayD::from_elem(IxDyn(&[1]), (step % 2) as f32);
                let result = env.step(&action);

                if let Some(render) = env.render() {
                    println!("Step {}: \n{}", step, render);
                }

                if result.done() {
                    tracing::info!(step, "Episode ended, resetting");
                    env.reset(None);
                }
            }
        }
        "squared" => {
            let mut env = Squared::new(2);
            env.reset(Some(42));

            for step in 0..steps {
                let action = ArrayD::from_elem(IxDyn(&[1]), (step % 8) as f32);
                let result = env.step(&action);

                if let Some(render) = env.render() {
                    println!("Step {}: \n{}", step, render);
                }

                if result.done() {
                    tracing::info!(step, "Episode ended, resetting");
                    env.reset(None);
                }
            }
        }
        _ => {
            tracing::error!("Unknown environment: {}", env_name);
        }
    }

    Ok(())
}

fn list_envs() {
    println!("Available environments:");
    println!();
    println!("  bandit     Multi-armed bandit (discrete, 4 arms)");
    println!("             Tests: basic optimization");
    println!();
    println!("  cartpole   CartPole classic control");
    println!("             Tests: continuous observation, discrete action");
    println!();
    println!("  squared    Grid navigation (5x5 grid)");
    println!("             Tests: spatial navigation");
    println!();
    println!("  memory     Sequence memorization");
    println!("             Tests: recurrent policies, credit assignment");
    println!();
    println!("Training requires --features torch and libtorch installed.");
}
#[cfg(feature = "torch")]
fn autotune(_env_name: &str, num_trials: usize, steps_per_trial: u64) -> Result<()> {
    use crate::hpo::{run_hpo_study, ParameterRange, SearchSpace};
    use pufferlib::policy::{MlpConfig, MlpPolicy};
    use pufferlib::vector::{Serial, VecEnv};
    use pufferlib_envs::Bandit;
    use std::collections::HashMap;

    let mut space = SearchSpace::new();
    space.add("learning_rate", ParameterRange::LogUniform(1e-4, 1e-2));
    space.add("vf_coef", ParameterRange::Uniform(0.1, 1.0));

    let result = run_hpo_study(
        "protein_autotune",
        space,
        num_trials,
        |trial_id, params| {
            let lr = *params.get("learning_rate").unwrap();
            let vf_coef = *params.get("vf_coef").unwrap();

            tracing::info!(trial = trial_id, lr, vf_coef, "Running HPO trial");

            let device = tch::Device::Cpu;
            let make_env = || Bandit::new(4);
            let envs = VecEnv::from_backend(Serial::new(make_env, 1));
            let obs_size = envs.observation_space().shape()[0];
            let num_actions = 4i64;

            let config = MlpConfig::default();
            let policy = MlpPolicy::new(obs_size as i64, num_actions, false, config, device);

            let trainer_config = pufferlib::training::TrainerConfig {
                total_timesteps: steps_per_trial,
                learning_rate: lr,
                vf_coef,
                ..Default::default()
            };

            let mut trainer = pufferlib::training::Trainer::new(envs, policy, trainer_config, device);
            trainer.trial_id = Some(trial_id);

            if let Err(e) = trainer.train() {
                tracing::error!(trial = trial_id, error = %e, "Trial failed");
                return -1e9;
            }

            let final_reward = trainer.reward();
            tracing::info!(trial = trial_id, reward = final_reward, "Trial completed");
            final_reward
        },
    );

    tracing::info!(
        best_reward = result.best_value,
        best_params = ?result.best_params,
        "Protein Auto-Tune Complete"
    );

    Ok(())
}
