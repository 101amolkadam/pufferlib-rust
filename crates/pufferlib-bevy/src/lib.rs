//! Bevy integration for PufferLib.
//!
//! Provides a plugin to bridge Bevy entities with PufferLib's RL loop.

pub use bevy;
use bevy::prelude::*;
use pufferlib::env::PufferEnv;

/// Plugin to enable RL support in Bevy
pub struct PufferRLPlugin;

/// Resource to store observations for all agents
#[derive(Resource, Debug, Clone, Default)]
pub struct AgentObservations {
    pub map: std::collections::HashMap<u32, Vec<f32>>,
}

/// Resource to store rewards for agents
#[derive(Resource, Debug, Clone, Default)]
pub struct Reward {
    pub map: std::collections::HashMap<u32, f32>,
}

/// Resource to store terminal status for agents
#[derive(Resource, Debug, Clone, Default)]
pub struct Done {
    pub map: std::collections::HashMap<u32, bool>,
}

/// Component to mark an entity as an RL agent
#[derive(Component, Debug, Clone)]
pub struct Agent {
    /// Unique identifier for the agent within the environment
    pub id: u32,
}

/// Bundle for RL-ready entities
#[derive(Bundle)]
pub struct AgentBundle {
    pub agent: Agent,
}

/// Event representing an action sent from PufferLib to a Bevy agent
#[derive(Event, Debug, Clone)]
pub struct AgentAction {
    pub agent_id: u32,
    pub data: Vec<f32>, // Flat action vector
}

/// Component to store the most recent observation for an agent
#[derive(Component, Debug, Clone, Default)]
pub struct Observation {
    pub data: Vec<f32>,
}

impl Plugin for PufferRLPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<AgentAction>();
        app.init_resource::<Reward>();
        app.init_resource::<Done>();
        app.init_resource::<AgentObservations>();
        app.add_systems(PostUpdate, sync_agent_observations);
    }
}

/// A PufferLib environment wrapping a Bevy application
pub struct PufferBevyEnv {
    /// The underlying Bevy app. Wrapped in a way to satisfy PufferEnv's Send bound.
    /// Note: User is responsible for ensuring the app is safe to move (usually true for headless).
    pub app: App,
    pub obs_space: pufferlib::spaces::DynSpace,
    pub act_space: pufferlib::spaces::DynSpace,
}

// Safety: Bevy App is generally not Send because of potential windowing/GUI handles.
// For headless RL use cases, we manually implement Send.
unsafe impl Send for PufferBevyEnv {}

impl PufferBevyEnv {
    pub fn new(
        mut app: App,
        obs_space: pufferlib::spaces::DynSpace,
        act_space: pufferlib::spaces::DynSpace,
    ) -> Self {
        if !app.is_plugin_added::<PufferRLPlugin>() {
            app.add_plugins(PufferRLPlugin);
        }
        Self {
            app,
            obs_space,
            act_space,
        }
    }
}

impl PufferEnv for PufferBevyEnv {
    fn observation_space(&self) -> pufferlib::spaces::DynSpace {
        self.obs_space.clone()
    }

    fn action_space(&self) -> pufferlib::spaces::DynSpace {
        self.act_space.clone()
    }

    fn reset(&mut self, _seed: Option<u64>) -> (ndarray::ArrayD<f32>, pufferlib::env::EnvInfo) {
        // One update to get initial obs
        self.app.update();
        let obs = self.collect_agent_obs(0);
        (obs, pufferlib::env::EnvInfo::default())
    }

    fn step(&mut self, action: &ndarray::ArrayD<f32>) -> pufferlib::env::StepResult {
        // 1. Inject actions
        {
            let mut action_events = self.app.world_mut().resource_mut::<Events<AgentAction>>();
            action_events.send(AgentAction {
                agent_id: 0,
                data: action.as_slice().unwrap().to_vec(),
            });
        }

        // 2. Update Bevy
        self.app.update();

        // 3. Extract results
        let observation = self.collect_agent_obs(0);
        let world = self.app.world();
        let reward = world
            .resource::<Reward>()
            .map
            .get(&0)
            .cloned()
            .unwrap_or(0.0);
        let done = world
            .resource::<Done>()
            .map
            .get(&0)
            .cloned()
            .unwrap_or(false);

        pufferlib::env::StepResult {
            observation,
            reward,
            terminated: done,
            truncated: false,
            info: pufferlib::env::EnvInfo::default(),
            cost: 0.0,
        }
    }
}

impl PufferBevyEnv {
    fn collect_agent_obs(&self, agent_id: u32) -> ndarray::ArrayD<f32> {
        let obs_data = self
            .app
            .world()
            .resource::<AgentObservations>()
            .map
            .get(&agent_id);
        if let Some(data) = obs_data {
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[data.len()]), data.clone()).unwrap()
        } else {
            let shape = self.obs_space.shape();
            ndarray::ArrayD::zeros(ndarray::IxDyn(&shape))
        }
    }
}

/// System to collect observations from agents
fn sync_agent_observations(
    query: Query<(&Agent, &Observation)>,
    mut obs_resource: ResMut<AgentObservations>,
) {
    for (agent, obs) in query.iter() {
        obs_resource.map.insert(agent.id, obs.data.clone());
    }
}
