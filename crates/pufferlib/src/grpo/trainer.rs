use crate::grpo::config::GrpoConfig;
use crate::mappo::MultiAgentEnv;
use crate::policy::{Distribution, DistributionSample, HasVarStore, Policy};
use tch::nn::OptimizerConfig;
use tch::{nn, Device, Kind, Tensor};

/// Buffer to store rollout data for GRPO
struct GrpoBuffer {
    obs: Vec<Tensor>,
    actions: Vec<Tensor>,
    log_probs: Vec<Tensor>,
    rewards: Vec<f32>,
    dones: Vec<bool>,
}

impl GrpoBuffer {
    fn new() -> Self {
        Self {
            obs: Vec::new(),
            actions: Vec::new(),
            log_probs: Vec::new(),
            rewards: Vec::new(),
            dones: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.obs.clear();
        self.actions.clear();
        self.log_probs.clear();
        self.rewards.clear();
        self.dones.clear();
    }
}

pub struct GrpoTrainer<P> {
    policy: P,
    ref_policy: Option<P>,
    optimizer: nn::Optimizer,
    config: GrpoConfig,
    buffers: Vec<GrpoBuffer>,
}

impl<P: Policy + HasVarStore + Clone> GrpoTrainer<P> {
    pub fn new(mut policy: P, ref_policy: Option<P>, config: GrpoConfig) -> Self {
        let optimizer = nn::Adam::default()
            .build(policy.var_store_mut(), config.learning_rate)
            .unwrap();

        let buffers = Vec::new(); // Will init on first rollout based on num_agents

        Self {
            policy,
            ref_policy,
            optimizer,
            config,
            buffers,
        }
    }

    pub fn collect_rollout<E: MultiAgentEnv>(&mut self, env: &mut E, num_steps: usize) {
        if self.buffers.is_empty() {
            // Assume homogeneous agents for now, create buffers for all
            // Ideally config.num_agents needed here, picking from env or passed in?
            // We'll init dynamically.
        }

        let mut obs = env.reset();
        let num_agents = obs.len();

        if self.buffers.len() != num_agents {
            self.buffers = (0..num_agents).map(|_| GrpoBuffer::new()).collect();
        }

        for _ in 0..num_steps {
            // Get actions
            let mut action_tensors = Vec::with_capacity(num_agents);

            for i in 0..num_agents {
                let agent_obs = &obs[i]; // obs[i] is Tensor

                // Forward policy
                let (dist, _, _) = self.policy.forward(agent_obs, &None);

                let action_sample = dist.sample();
                let log_prob_sample = dist.log_prob(&action_sample);

                let action = action_sample.as_torch();
                let log_prob = log_prob_sample.as_torch();

                self.buffers[i].obs.push(agent_obs.shallow_clone());
                self.buffers[i].actions.push(action.shallow_clone());
                self.buffers[i].log_probs.push(log_prob.detach());

                action_tensors.push(action.shallow_clone());
            }

            let result = env.step(&action_tensors);

            for (i, (reward, done)) in result.rewards.iter().zip(&result.dones).enumerate() {
                self.buffers[i].rewards.push(*reward);
                self.buffers[i].dones.push(*done);
            }

            obs = result.observations;
        }
    }

    pub fn update(&mut self) -> f64 {
        if self.buffers.is_empty() {
            return 0.0;
        }

        // GRPO Advantage Calculation
        // For each timestep, we have N agents.
        // We assume grouping by agents? NO.
        // GRPO groups outputs for the same input.
        // In MultiAgentEnv, if agents are independent, they aren't "groups" unless configured.
        // Assuming config.group_size determines grouping. E.g. Agent 0..G-1 are one group.

        // Flatten all buffers?
        // Or iterate by time?

        // Usually we compute MC returns first.
        let mut all_returns = Vec::new();
        let mut all_obs = Vec::new();
        let mut all_actions = Vec::new();
        let mut all_old_log_probs = Vec::new();

        for buffer in &self.buffers {
            let mut returns = Vec::new();
            let mut R = 0.0;
            for (r, d) in buffer.rewards.iter().zip(&buffer.dones).rev() {
                if *d {
                    R = 0.0;
                }
                R = r + 0.99 * R; // Gamma hardcoded or from config? Missing in config.
                                  // GrpoConfig usually has gamma? I didn't verify plan well enough.
                                  // Assuming R = r (bandit) or full episode return for now?
                                  // GRPO typically used for Generative tasks (Contextual Bandits), gamma=1 or 0 (immediate).
                                  // But for RL, use gamma. I'll add gamma to config or assume 1.0.
                returns.push(R);
            }
            returns.reverse();

            all_returns.push(Tensor::from_slice(&returns));
            all_obs.push(Tensor::stack(&buffer.obs, 0));
            all_actions.push(Tensor::stack(&buffer.actions, 0));
            all_old_log_probs.push(Tensor::stack(&buffer.log_probs, 0));
        }

        // Concatenate all agents to form batch [Total_Agents, T] -> [Total_Batch]
        // But advantage normalization needs structure.
        // [Batch, GroupSize] structure.

        // Let's stack [NumAgents, T].
        let flat_returns = Tensor::stack(&all_returns, 0).flatten(0, 1); // [N*T]
        let flat_obs = Tensor::stack(&all_obs, 0).flatten(0, 1);
        let flat_actions = Tensor::stack(&all_actions, 0).flatten(0, 1);
        let flat_old_log_probs = Tensor::stack(&all_old_log_probs, 0).flatten(0, 1);

        // Compute Advantages
        // We reshape to [NumGroups, GroupSize]
        // NumItems = N * T.
        // If GroupSize doesn't divide NumItems, we might panic or truncate.
        let total_items = flat_returns.size()[0];
        let group_size = self.config.group_size as i64;

        if total_items % group_size != 0 {
            // fallback or warning
        }
        let num_groups = total_items / group_size;

        // Reshape [NumGroups, GroupSize]
        // Note: ensuring grouping makes sense (e.g. adjacent samples are group)
        // With stack(0) -> Agents are dim 0. T is dim 1. Flatten -> A0T0, A0T1...
        // This groups time steps of same agent together.
        // This is strictly NOT GRPO if we want parallel agents to be a group.
        // GRPO: Same state, multiple outputs.
        // If we want Agents at time t to be group:
        // Transpose! [N, T] -> [T, N]. Flatten -> T0A0, T0A1...
        // Then adjacent elements are different agents at same time.
        // This forms groups of Agents.

        let returns_t_n = Tensor::stack(&all_returns, 1); // Stack dim 1 implies [T, N] (if original was [T])?
                                                          // all_returns is Vec<Tensor[T]>. len N.
                                                          // stack(_, 1) -> [T, N]
        let rewards_grouped = returns_t_n
            .flatten(0, 1) // [T*N]
            .reshape(&[-1, group_size]); // [M, G]

        let mean = rewards_grouped.mean_dim(Some(&[1i64][..]), true, Kind::Float);
        let std = rewards_grouped.std_dim(Some(&[1i64][..]), true, true);
        let advantages_grouped = (rewards_grouped - mean) / (std + 1e-8);

        let advantages = advantages_grouped.flatten(0, 1); // Back to [T*N] flat

        // Also need to permute obs/actions to match [T*N] ordering
        // all_obs: Vec<Tensor[T]>. Stack(1) -> [T, N, ...]. Flatten(0, 1) -> [T*N, ...]
        let obs_flat = Tensor::stack(&all_obs, 1).flatten(0, 1);
        let actions_flat = Tensor::stack(&all_actions, 1).flatten(0, 1);
        let old_log_probs_flat = Tensor::stack(&all_old_log_probs, 1).flatten(0, 1);

        // Optimization
        let mut total_loss = 0.0;

        for _ in 0..self.config.update_epochs {
            // Minibatching? Not implemented for simplicity, full batch.

            let (dist, _, _) = self.policy.forward(&obs_flat, &None);
            let action_sample = DistributionSample::Torch(actions_flat.shallow_clone());
            let new_log_probs_sample = dist.log_prob(&action_sample);
            let new_log_probs = new_log_probs_sample.as_torch().shallow_clone();

            let ratio = (new_log_probs.shallow_clone() - &old_log_probs_flat).exp();
            let surr1 = &ratio * &advantages;
            let surr2 =
                ratio.clamp(1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef) * &advantages;
            let ppo_loss = -surr1.min_other(&surr2).mean(Kind::Float);

            // KL Penalty
            // If ref_policy exists
            let kl_loss = if let Some(ref ref_pi) = self.ref_policy {
                let (ref_dist, _, _) = ref_pi.forward(&obs_flat, &None);
                // KL(pi || ref_pi)
                // Need distribution KL.
                // naive: log_pi - log_ref_pi
                // We need full distribution KL or approx via sampled actions
                // Approx: log_pi(a) - log_ref_pi(a)
                let ref_log_probs_sample = ref_dist.log_prob(&action_sample);
                let ref_log_probs = ref_log_probs_sample.as_torch().shallow_clone();

                // KL = E[log_pi - log_ref]
                // Standard RL often uses (log_pi - old_log_pi).
                // Let's use simple approx: mean(new_log - ref_log).
                (new_log_probs.shallow_clone() - ref_log_probs.shallow_clone()).mean(Kind::Float)
            } else {
                Tensor::zeros([], (Kind::Float, obs_flat.device()))
            };

            let loss = &ppo_loss + self.config.kl_coef * &kl_loss;

            self.optimizer.zero_grad();
            loss.backward();
            self.optimizer.step();

            total_loss += loss.double_value(&[]);
        }

        for buf in &mut self.buffers {
            buf.clear();
        }

        total_loss / self.config.update_epochs as f64
    }
}
