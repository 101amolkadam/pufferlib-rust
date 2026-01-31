//! Tonics-based RPC for PufferLib.

pub mod remote {
    tonic::include_proto!("pufferlib_rpc");
}

use ndarray::{ArrayD, IxDyn};
use pufferlib::env::{EnvInfo, PufferEnv, StepResult};
use remote::observation_service_client::ObservationServiceClient;
use remote::{ResetRequest, StepRequest};
use tonic::transport::Channel;

/// A PufferLib environment that communicates with a remote server over gRPC
pub struct RemoteEnv {
    client: ObservationServiceClient<Channel>,
    obs_space: pufferlib::spaces::DynSpace,
    act_space: pufferlib::spaces::DynSpace,
    rt: tokio::runtime::Runtime,
}

impl RemoteEnv {
    pub fn new(
        address: String,
        obs_space: pufferlib::spaces::DynSpace,
        act_space: pufferlib::spaces::DynSpace,
    ) -> Self {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let client =
            rt.block_on(async { ObservationServiceClient::connect(address).await.unwrap() });

        Self {
            client,
            obs_space,
            act_space,
            rt,
        }
    }
}

impl PufferEnv for RemoteEnv {
    fn observation_space(&self) -> pufferlib::spaces::DynSpace {
        self.obs_space.clone()
    }

    fn action_space(&self) -> pufferlib::spaces::DynSpace {
        self.act_space.clone()
    }

    fn reset(&mut self, seed: Option<u64>) -> (ArrayD<f32>, EnvInfo) {
        let response = self.rt.block_on(async {
            self.client
                .reset(ResetRequest {
                    seed: seed.unwrap_or(0),
                })
                .await
                .unwrap()
        });

        let obs = response.into_inner().observation;
        (
            ArrayD::from_shape_vec(IxDyn(&[obs.len()]), obs).unwrap(),
            EnvInfo::default(),
        )
    }

    fn step(&mut self, action: &ArrayD<f32>) -> StepResult {
        let response = self.rt.block_on(async {
            self.client
                .step(StepRequest {
                    action: action.as_slice().unwrap().to_vec(),
                })
                .await
                .unwrap()
        });

        let resp = response.into_inner();
        StepResult {
            observation: ArrayD::from_shape_vec(IxDyn(&[resp.observation.len()]), resp.observation)
                .unwrap(),
            reward: resp.reward,
            terminated: resp.terminated,
            truncated: resp.truncated,
            info: EnvInfo::default(),
            cost: resp.cost,
        }
    }
}

// Safety: RemoteEnv uses a Local Runtime and Channel, we must manually ensure Send if needed.
// However, since we block on the runtime for every call, it's effectively synchronous once initialized.
unsafe impl Send for RemoteEnv {}
