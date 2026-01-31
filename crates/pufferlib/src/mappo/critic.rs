//! Centralized Critic implementation.

use tch::{nn, Device, Tensor};

/// Centralized critic using global state
pub struct CentralizedCritic {
    net: nn::Sequential,
    vs: nn::VarStore,
}

impl CentralizedCritic {
    pub fn new(global_state_dim: i64, hidden_dim: i64, device: Device) -> Self {
        let vs = nn::VarStore::new(device);
        let net = nn::seq()
            .add(nn::linear(
                &vs.root() / "fc1",
                global_state_dim,
                hidden_dim,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                &vs.root() / "fc2",
                hidden_dim,
                hidden_dim,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                &vs.root() / "fc3",
                hidden_dim,
                1,
                Default::default(),
            ));

        Self { net, vs }
    }

    pub fn forward(&self, global_state: &Tensor) -> Tensor {
        global_state.apply(&self.net)
    }

    pub fn var_store_mut(&mut self) -> &mut nn::VarStore {
        &mut self.vs
    }
}
