use pufferlib::policy::{Distribution, HasVarStore, Policy};
use pufferlib::training::{Trainer, TrainerConfig};
use pufferlib::vector::{Parallel, VecEnv};
use pufferlib_envs::CartPole;
use tch::{nn, Device, Kind, Tensor};

struct SimplePolicy {
    vs: nn::VarStore,
}

impl SimplePolicy {
    fn new(device: Device) -> Self {
        let mut vs = nn::VarStore::new(device);
        let root = vs.root();
        let _ = root.zeros("weight", &[1]);
        Self { vs }
    }
}

impl Policy for SimplePolicy {
    fn forward(
        &self,
        obs: &Tensor,
        _state: &Option<Vec<Tensor>>,
    ) -> (Distribution, Tensor, Option<Vec<Tensor>>) {
        let batch_size = obs.size()[0];
        // Dummy dependency on weights to ensure backward() works
        let _dummy = self.vs.root().get("weight").unwrap().sum(Kind::Float);
        let values =
            Tensor::zeros([batch_size], (Kind::Float, obs.device())) + _dummy.shallow_clone() * 0.0;
        let logits = Tensor::zeros([batch_size, 2], (Kind::Float, obs.device())) + _dummy * 0.0;
        (Distribution::Categorical { logits }, values, None)
    }

    fn initial_state(&self, _batch_size: i64) -> Option<Vec<Tensor>> {
        None
    }
}

impl HasVarStore for SimplePolicy {
    fn var_store(&self) -> &nn::VarStore {
        &self.vs
    }
    fn var_store_mut(&mut self) -> &mut nn::VarStore {
        &mut self.vs
    }
}

fn main() {
    let device = Device::Cpu;
    let env_factory = || CartPole::new();
    let backend = Parallel::new(env_factory, 4);
    let vecenv = VecEnv::from_backend(backend);

    let policy = SimplePolicy::new(device);
    let config = TrainerConfig::default().with_timesteps(1000);

    let mut trainer =
        Trainer::<SimplePolicy, _, pufferlib::training::optimizer::TorchOptimizer>::new(
            vecenv, policy, config, device,
        );
    if let Err(e) = trainer.train() {
        eprintln!("Training failed with error: {:?}", e);
    } else {
        println!("Success!");
    }
}
