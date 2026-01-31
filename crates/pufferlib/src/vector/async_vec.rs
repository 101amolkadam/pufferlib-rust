use crate::env::EnvInfo;
use crate::spaces::DynSpace;
use crate::types::Vec;
use crate::vector::{ObservationBatch, VecEnvBackend, VecEnvResult};
use crossbeam_channel::{bounded, Receiver, Sender};
use ndarray::Array2;

#[cfg(feature = "std")]
use std::marker::PhantomData;
#[cfg(feature = "std")]
use std::thread::{spawn, JoinHandle};

enum Command {
    Reset(Option<u64>),
    Step(Array2<f32>),
    Close,
}

enum Response {
    Reset(ObservationBatch, Vec<EnvInfo>),
    Step(VecEnvResult),
}

/// Async vectorized environment backend
#[cfg(feature = "std")]
pub struct AsyncVecEnv<B: VecEnvBackend + 'static> {
    cmd_tx: Sender<Command>,
    res_rx: Receiver<Response>,
    _worker: Option<JoinHandle<()>>,
    obs_space: DynSpace,
    action_space: DynSpace,
    num_envs: usize,
    _phantom: PhantomData<B>,
}

#[cfg(feature = "std")]
impl<B: VecEnvBackend + 'static> AsyncVecEnv<B> {
    /// Create a new async backend wrapping another backend
    pub fn new(mut backend: B) -> Self {
        let obs_space = backend.observation_space();
        let action_space = backend.action_space();
        let num_envs = backend.num_envs();

        let (cmd_tx, cmd_rx) = bounded(1);
        let (res_tx, res_rx) = bounded(1);

        let worker = spawn(move || {
            while let Ok(cmd) = cmd_rx.recv() {
                match cmd {
                    Command::Reset(seed) => {
                        let res = backend.reset(seed);
                        if res_tx.send(Response::Reset(res.0, res.1)).is_err() {
                            break;
                        }
                    }
                    Command::Step(actions) => {
                        let res = backend.step(&actions);
                        if res_tx.send(Response::Step(res)).is_err() {
                            break;
                        }
                    }
                    Command::Close => {
                        backend.close();
                        break;
                    }
                }
            }
        });

        Self {
            cmd_tx,
            res_rx,
            _worker: Some(worker),
            obs_space,
            action_space,
            num_envs,
            _phantom: PhantomData,
        }
    }
}

#[cfg(feature = "std")]
impl<B: VecEnvBackend + 'static> VecEnvBackend for AsyncVecEnv<B> {
    fn observation_space(&self) -> DynSpace {
        self.obs_space.clone()
    }

    fn action_space(&self) -> DynSpace {
        self.action_space.clone()
    }

    fn num_envs(&self) -> usize {
        self.num_envs
    }

    fn reset(&mut self, seed: Option<u64>) -> (ObservationBatch, Vec<EnvInfo>) {
        self.cmd_tx.send(Command::Reset(seed)).unwrap();
        match self.res_rx.recv().unwrap() {
            Response::Reset(obs, infos) => (obs, infos),
            _ => panic!("Expected Reset response"),
        }
    }

    fn step(&mut self, actions: &Array2<f32>) -> VecEnvResult {
        // This is still blocking for now, but the worker thread
        // allows the main thread to do other things if we use a
        // different API or if we use this for distributed workers.
        self.cmd_tx.send(Command::Step(actions.clone())).unwrap();
        match self.res_rx.recv().unwrap() {
            Response::Step(result) => result,
            _ => panic!("Expected Step response"),
        }
    }

    fn close(&mut self) {
        let _ = self.cmd_tx.send(Command::Close);
        if let Some(worker) = self._worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(feature = "std")]
impl<B: VecEnvBackend + 'static> Drop for AsyncVecEnv<B> {
    fn drop(&mut self) {
        self.close();
    }
}
