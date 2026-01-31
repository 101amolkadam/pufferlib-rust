use crate::dreamer::config::DreamerConfig;
use crate::dreamer::models::{DecoderCNN, DenseHead, EncoderCNN};
use crate::dreamer::rssm::{State, RSSM};
use tch::nn::OptimizerConfig;
use tch::{nn, Device, Kind, Tensor};

pub struct DreamerTrainer {
    config: DreamerConfig,
    // World Model
    rssm: RSSM,
    encoder: EncoderCNN,
    decoder: DecoderCNN,
    reward_head: DenseHead,
    continue_head: DenseHead,

    // Actor Critic
    actor: DenseHead,  // Simple MLP actor for now
    critic: DenseHead, // Value function

    // Optims
    model_opt: nn::Optimizer,
    actor_opt: nn::Optimizer,
    value_opt: nn::Optimizer,
}

impl DreamerTrainer {
    pub fn new(
        vs: &nn::VarStore,
        config: DreamerConfig,
        obs_channels: i64,
        action_dim: i64,
    ) -> Self {
        let p = &vs.root();
        let encoder = EncoderCNN::new(&(p / "encoder"), obs_channels);
        let decoder = DecoderCNN::new(
            &(p / "decoder"),
            config.deter_size + config.stoch_size * config.stoch_discrete,
            [obs_channels, 64, 64],
        );

        // Heads inputs are deter + stoch
        let feature_dim = config.deter_size + config.stoch_size * config.stoch_discrete;

        let reward_head = DenseHead::new(&(p / "reward"), feature_dim, 1, 2, 256);
        let continue_head = DenseHead::new(&(p / "continue"), feature_dim, 1, 2, 256);

        let rssm = RSSM::new(&(p / "rssm"), config.clone(), action_dim);

        let actor = DenseHead::new(&(p / "actor"), feature_dim, action_dim, 3, 256);
        let critic = DenseHead::new(&(p / "critic"), feature_dim, 1, 3, 256);

        let model_opt = nn::Adam::default()
            .build(vs, config.model_lr)
            .expect("Failed to build opt");
        let actor_opt = nn::Adam::default()
            .build(vs, config.actor_lr)
            .expect("Failed to build opt");
        let value_opt = nn::Adam::default()
            .build(vs, config.value_lr)
            .expect("Failed to build opt");

        Self {
            config,
            rssm,
            encoder,
            decoder,
            reward_head,
            continue_head,
            actor,
            critic,
            model_opt,
            actor_opt,
            value_opt,
        }
    }

    /// Update World Model (Representation Learning)
    pub fn update_model(
        &mut self,
        obs: &Tensor,
        actions: &Tensor,
        rewards: &Tensor,
        dones: &Tensor,
    ) -> f64 {
        // 1. Embed Observations
        // obs: [B, T, C, H, W]
        let s = obs.size();
        let (b, t, c, h, w) = (s[0], s[1], s[2], s[3], s[4]);
        let obs_flat = obs.reshape(&[b * t, c, h, w]);
        let embed = self.encoder.forward(&obs_flat).reshape(&[b, t, -1]);

        // 2. RSSM Observe Rollout
        // Need to loop over T steps or use sequence processing
        // Init state
        let device = obs.device();
        let mut state = self.rssm.initial_state(b, device);

        let mut priors = Vec::new();
        let mut posteriors = Vec::new();

        for i in 0..t {
            let embed_t = embed.get(i);
            let action_t = actions.get(i);
            // State update
            let (post, prior) = self.rssm.observe(&embed_t, &action_t, &state);
            priors.push(prior);
            posteriors.push(post.clone());
            state = post;
        }

        // 3. Loss Calculation (Reconstruction, Reward, KL)
        // Flatten posteriors
        // let post_features = cat(deter + stoch)
        // recon = decoder(post_features)
        // dist = MSE(recon, obs)
        // kl = KL(post || prior)
        // reward_pred = ...

        // Placeholder
        0.0
    }

    /// Update Actor Critic (Behavior Learning) using imagined trajectories
    pub fn update_actor_critic(&mut self, start_state: State) -> f64 {
        let (b, _deter_size) = start_state.deter.size2().unwrap();

        // 1. Imagine trajectories
        let mut states = Vec::new();
        let mut actions = Vec::new();

        // Use no_grad for world model during imagination if needed,
        // but here we just process
        let mut current_state = start_state;

        for _ in 0..self.config.horizon {
            let feat = current_state.get_features();
            // Actor: predict action from latent features
            let action_logits = self.actor.forward(&feat);
            // Sample action
            let action = action_logits.tanh(); // Simple Tanh for continuous action space

            states.push(current_state.clone());
            actions.push(action.shallow_clone());

            // Imagine next latent state
            current_state = self.rssm.imagine(&action, &current_state);
        }

        // 2. Compute Rewards and Continues for imagined states
        let mut rewards = Vec::new();
        let mut continues = Vec::new();
        let mut values = Vec::new();

        for s in &states {
            let feat = s.get_features();
            rewards.push(self.reward_head.forward(&feat));
            continues.push(self.continue_head.forward(&feat).sigmoid());
            values.push(self.critic.forward(&feat));
        }

        // 3. Compute Value Targets (Lambda-returns)
        let rewards = Tensor::stack(&rewards, 0);
        let values = Tensor::stack(&values, 0);
        let continues = Tensor::stack(&continues, 0);

        let last_feat = current_state.get_features();
        let last_value = self.critic.forward(&last_feat);

        let targets = compute_lambda_returns(&rewards, &values, &continues, &last_value, 0.95);

        // 4. Update Actor (Maximize reward + value target)
        self.actor_opt.zero_grad();
        let actor_loss = -targets.mean(Kind::Float);
        actor_loss.backward();
        self.actor_opt.step();

        // 5. Update Critic (Fit value function to targets)
        self.value_opt.zero_grad();
        let value_loss = (values - targets.detach())
            .pow_tensor_scalar(2)
            .mean(Kind::Float);
        value_loss.backward();
        self.value_opt.step();

        actor_loss.double_value(&[])
    }
}

/// Compute Lambda-returns for imagination horizon
fn compute_lambda_returns(
    rewards: &Tensor,
    values: &Tensor,
    continues: &Tensor,
    bootstrap: &Tensor,
    lambda: f64,
) -> Tensor {
    let t = rewards.size()[0];
    let mut targets = Vec::with_capacity(t as usize);
    let mut next_val = bootstrap.shallow_clone();

    // Default gamma
    let gamma = 0.99;

    for i in (0..t).rev() {
        let r = rewards.get(i);
        let v = values.get(i);
        let c = continues.get(i);

        // V_lambda(t) = r(t) + gamma * continue(t) * [(1-lambda) * v(t+1) + lambda * V_lambda(t+1)]
        next_val = &r + gamma * &c * ((1.0 - lambda) * &v + lambda * &next_val);
        targets.push(next_val.shallow_clone());
    }
    targets.reverse();
    Tensor::stack(&targets, 0)
}
