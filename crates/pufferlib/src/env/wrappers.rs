//! Environment wrappers for common functionality.

use super::{EnvInfo, PufferEnv, StepResult};
use crate::spaces::DynSpace;
use ndarray::ArrayD;

/// Wrapper that tracks episode statistics (return and length).
///
/// Adds `episode_return` and `episode_length` to info on episode completion.
pub struct EpisodeStats<E: PufferEnv> {
    env: E,
    episode_return: f32,
    episode_length: u32,
}

impl<E: PufferEnv> EpisodeStats<E> {
    /// Wrap an environment with episode statistics tracking
    pub fn new(env: E) -> Self {
        Self {
            env,
            episode_return: 0.0,
            episode_length: 0,
        }
    }

    /// Get a reference to the inner environment
    pub fn inner(&self) -> &E {
        &self.env
    }

    /// Get a mutable reference to the inner environment
    pub fn inner_mut(&mut self) -> &mut E {
        &mut self.env
    }
}

impl<E: PufferEnv> PufferEnv for EpisodeStats<E> {
    fn observation_space(&self) -> DynSpace {
        self.env.observation_space()
    }

    fn action_space(&self) -> DynSpace {
        self.env.action_space()
    }

    fn reset(&mut self, seed: Option<u64>) -> (ArrayD<f32>, EnvInfo) {
        self.episode_return = 0.0;
        self.episode_length = 0;
        self.env.reset(seed)
    }

    fn step(&mut self, action: &ArrayD<f32>) -> StepResult {
        let mut result = self.env.step(action);

        self.episode_return += result.reward;
        self.episode_length += 1;

        if result.done() {
            result.info = result
                .info
                .with_episode_stats(self.episode_return, self.episode_length);

            // Reset internal counters (env will be reset externally)
            self.episode_return = 0.0;
            self.episode_length = 0;
        }

        result
    }

    fn render(&self) -> Option<String> {
        self.env.render()
    }

    fn close(&mut self) {
        self.env.close()
    }

    fn num_agents(&self) -> usize {
        self.env.num_agents()
    }
}

/// Wrapper that clips continuous actions to the action space bounds.
pub struct ClipAction<E: PufferEnv> {
    env: E,
    low: ArrayD<f32>,
    high: ArrayD<f32>,
}

impl<E: PufferEnv> ClipAction<E> {
    /// Wrap an environment with action clipping
    ///
    /// Requires the action space to be a Box space.
    pub fn new(env: E) -> Self {
        let space = env.action_space();
        let (low, high) = match &space {
            DynSpace::Box(b) => (b.low.clone(), b.high.clone()),
            _ => panic!("ClipAction requires a Box action space"),
        };
        Self { env, low, high }
    }
}

impl<E: PufferEnv> PufferEnv for ClipAction<E> {
    fn observation_space(&self) -> DynSpace {
        self.env.observation_space()
    }

    fn action_space(&self) -> DynSpace {
        self.env.action_space()
    }

    fn reset(&mut self, seed: Option<u64>) -> (ArrayD<f32>, EnvInfo) {
        self.env.reset(seed)
    }

    fn step(&mut self, action: &ArrayD<f32>) -> StepResult {
        // Clip action to bounds

        let mut clipped = action.clone();
        for ((a, &l), &h) in clipped
            .iter_mut()
            .zip(self.low.iter())
            .zip(self.high.iter())
        {
            *a = a.max(l).min(h);
        }

        self.env.step(&clipped)
    }

    fn render(&self) -> Option<String> {
        self.env.render()
    }

    fn close(&mut self) {
        self.env.close()
    }

    fn num_agents(&self) -> usize {
        self.env.num_agents()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::{Box as BoxSpace, Discrete};
    use ndarray::IxDyn;

    // Simple test environment
    struct SimpleEnv {
        step_count: u32,
    }

    impl PufferEnv for SimpleEnv {
        fn observation_space(&self) -> DynSpace {
            DynSpace::Box(BoxSpace::uniform(&[2], 0.0, 1.0))
        }

        fn action_space(&self) -> DynSpace {
            DynSpace::Discrete(Discrete::new(2))
        }

        fn reset(&mut self, _seed: Option<u64>) -> (ArrayD<f32>, EnvInfo) {
            self.step_count = 0;
            (ArrayD::zeros(IxDyn(&[2])), EnvInfo::new())
        }

        fn step(&mut self, _action: &ArrayD<f32>) -> StepResult {
            self.step_count += 1;
            StepResult {
                observation: ArrayD::zeros(IxDyn(&[2])),
                reward: 1.0,
                terminated: self.step_count >= 5,
                truncated: false,
                info: EnvInfo::new(),
            }
        }
    }

    #[test]
    fn test_episode_stats() {
        let env = SimpleEnv { step_count: 0 };
        let mut wrapped = EpisodeStats::new(env);

        wrapped.reset(None);

        let action = ArrayD::zeros(IxDyn(&[1]));
        for _ in 0..4 {
            let result = wrapped.step(&action);
            assert!(!result.done());
            assert!(result.info.get("episode_return").is_none());
        }

        // 5th step should terminate
        let result = wrapped.step(&action);
        assert!(result.done());
        assert_eq!(result.info.get("episode_return"), Some(5.0));
        assert_eq!(result.info.get("episode_length"), Some(5.0));
    }
}
