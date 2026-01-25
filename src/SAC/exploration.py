import numpy as np
import torch as T
import torch.nn as nn
import torch.autograd

# Exploration Strategies for Continuous Action Spaces

class ExplorerStrategy:
    """
    Base class for exploration strategies.
    """
    def __init__(self, n_actions: int, low=-1, high=1, name: str = "base-explorer"):
        self.n_actions = n_actions
        self.low = low
        self.high = high
        self.name = name
        self.num_envs = 1
        self.supports_vec_env = False

    def choose_action(self, observation, agent=None, step=None, warmup=False):
        raise NotImplementedError

    def get_intrinsic_reward(self, state, action, next_state):
        return 0.0

    def update(self, *args, **kwargs):
        pass

    def reset(self):
        pass

    def end_episode(self):
        pass

    def id(self):
        return self.name

    def __str__(self):
        return self.name

    def support_vec_env(self, num_envs: int):
        self.num_envs = num_envs
        self.supports_vec_env = True


class RandomExplorer(ExplorerStrategy):
    """
    Random (Uniform) exploration strategy.
    """
    def __init__(self, n_actions: int, low=-1, high=1, name: str = "uniform-explorer"):
        super().__init__(n_actions, low, high, name)

    def choose_action(self, observation, agent=None, step=None, warmup=False):
        if warmup:
            return np.random.uniform(low=self.low, high=self.high, size=self.n_actions)

        # use agent's policy, the alpha-scaled entropy should be good enough
        return agent.choose_action_from_policy(observation)

    def choose_action_batch(self, observations, agent=None, step=None, warmup=False):
        if warmup:
            return np.random.uniform(low=self.low, high=self.high, size=(self.num_envs, self.n_actions))

        return agent.choose_action_from_policy_batch(observations)

    def id(self):
        return "uniform_explorer"

    def __str__(self):
        return f"{self.name} ({self.n_actions} actions)"


class GaussianExplorer(ExplorerStrategy):
    """
    Gaussian exploration strategy.
    """
    def __init__(self, n_actions: int, mu=0.0, sigma=0.8, low=-1, high=1, name: str = "gaussian-explorer"):
        super().__init__(n_actions, low, high, name)
        self.mu = mu
        self.sigma = sigma

    def get_noise(self):
        out_l = np.random.normal(loc=-1 * self.mu, scale=self.sigma, size=self.n_actions)
        out_r = np.random.normal(loc=self.mu, scale=self.sigma, size=self.n_actions)
        noise = np.where(np.random.rand(self.n_actions) < 0.5, out_l, out_r)
        return noise

    def get_noise_batch(self):
        out_l = np.random.normal(loc=-1 * self.mu, scale=self.sigma, size=(self.num_envs, self.n_actions))
        out_r = np.random.normal(loc=self.mu, scale=self.sigma, size=(self.num_envs, self.n_actions))
        rand_vals = np.random.rand(self.num_envs, self.n_actions)
        noise = np.where(rand_vals < 0.5, out_l, out_r)
        return noise

    def choose_action(self, observation, agent=None, step=None, warmup=False):
        if warmup:
            return np.clip(self.get_noise(), self.low, self.high)

        return agent.choose_action_from_policy(observation)

    def choose_action_batch(self, observations, agent=None, step=None, warmup=False):
        if warmup:
            return np.clip(self.get_noise_batch(), self.low, self.high)
        return agent.choose_action_from_policy_batch(observations)

    def id(self):
        return f"gaussian_explorer_sigma_{self.sigma:.2f}"

    def __str__(self):
        return f"{self.name} (sigma={self.sigma})"


class OrnsteinUhlenbeckExplorer(ExplorerStrategy):
    """
    Ornstein-Uhlenbeck exploration strategy.
    """
    def __init__(self, n_actions: int, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2, x0=None,
                 max_episodes=int(1e5 // 200), low=-1, high=1, name: str = "ou-explorer"):
        super().__init__(n_actions, low, high, name)
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.init_sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.max_episodes = max_episodes
        self.x_prev = np.zeros(self.n_actions) if x0 is None else x0
        self.reset()

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.n_actions)

    def noise(self):
        x = (self.x_prev +
             self.theta * (self.mu - self.x_prev) * self.dt +
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.n_actions))
        self.x_prev = x
        return x

    def noise_batch(self):
        x = (self.x_prev +
             self.theta * (self.mu - self.x_prev) * self.dt +
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=(self.num_envs, self.n_actions)))
        self.x_prev = x
        return x

    def choose_action(self, observation, agent=None, step=None, warmup=False):
        ou_noise = self.noise()

        if warmup:
            return np.clip(ou_noise, self.low, self.high)

        state_tensor = T.from_numpy(observation).float().unsqueeze(0).to(agent.actor.device)
        with T.no_grad():
            mu, _ = agent.actor.sample_normal(state_tensor, reparameterize=False)
            action_mean = mu.cpu().detach().numpy()[0]

        action = action_mean + ou_noise
        return np.clip(action, self.low, self.high)

    def choose_action_batch(self, observations, agent=None, step=None, warmup=False):
        ou_noise = self.noise_batch()

        if warmup:
            return np.clip(ou_noise, self.low, self.high)

        state_tensor = T.from_numpy(observations).float().to(agent.actor.device)
        with T.no_grad():
            mu, _ = agent.actor.sample_normal(state_tensor, reparameterize=False)
            action_mean = mu.cpu().detach().numpy()

        action = action_mean + ou_noise
        return np.clip(action, self.low, self.high)

    def end_episode(self):
        self.sigma = self.sigma * min((1 - 1 / self.max_episodes), 0.99)

    def id(self):
        return f"ou_explorer_theta_{self.theta:.2f}_sigma_{self.init_sigma:.2f}"

    def __str__(self):
        return f"{self.name} (theta={self.theta}, sigma={self.init_sigma})"


class OptimisticExplorer(ExplorerStrategy):
    """
    Optimistic Actor Critic (OAC) Approximation.
    Uses Upper Confidence Bound of Q-values to guide exploration.
    Ref: Better Exploration with Optimistic Actor-Critic (Ciosek et al., 2020)
    """
    def __init__(self, agent, sub_cfg, low=-1, high=1, name: str = "optimistic-explorer"):
        super().__init__(agent.cfg.n_actions, low, high, name)
        self.beta = sub_cfg.beta

    def choose_action(self, observation, agent=None, step=None, warmup=False):
        if warmup:
             return np.random.uniform(low=self.low, high=self.high, size=self.n_actions)

        state_tensor = T.tensor(observation, dtype=T.float32).unsqueeze(0).to(agent.actor.device)
        with T.no_grad():
            mux, stdx = agent.actor.forward(state_tensor)
        mux.requires_grad_(True)
        q1 = agent.critic_1.forward(state_tensor, mux)
        q2 = agent.critic_2.forward(state_tensor, mux)
        mean_q = (q1 + q2) / 2.0
        std_q = T.abs(q1 - q2) / 2.0
        q_ub = mean_q + self.beta * std_q

        q_ub.backward()
        grad_mu = mux.grad

        # mu_opt = mux + k_UB * Sigma_pi * grad_Q (approx)
        if grad_mu is not None:
             mu_opt = mux + self.beta * stdx * grad_mu
        else:
             mu_opt = mu
        mu_opt = mu_opt.detach()

        action = T.normal(mu_opt, stdx)
        action = T.tanh(action)
        action = action.squeeze(0).cpu().detach().numpy()
        return np.clip(action, self.low, self.high)

    def choose_action_batch(self, observations, agent=None, step=None, warmup=False):
        if warmup:
             return np.random.uniform(low=self.low, high=self.high, size=(self.num_envs, self.n_actions))

        state_tensor = T.from_numpy(observations).float().to(agent.actor.device)
        with T.no_grad():
            mux, stdx = agent.actor.forward(state_tensor)
        mux.requires_grad_(True)
        q1 = agent.critic_1.forward(state_tensor, mux)
        q2 = agent.critic_2.forward(state_tensor, mux)
        mean_q = (q1 + q2) / 2.0
        std_q = T.abs(q1 - q2) / 2.0
        q_ub = mean_q + self.beta * std_q

        q_ub.backward(T.ones_like(q_ub))
        grad_mu = mux.grad

        # mu_opt = mux + k_UB * Sigma_pi * grad_Q (approx)
        if grad_mu is not None:
             mu_opt = mux + self.beta * stdx * grad_mu
        else:
             mu_opt = mu
        mu_opt = mu_opt.detach()

        action = T.normal(mu_opt, stdx)
        action = T.tanh(action)
        action = action.cpu().detach().numpy()
        return np.clip(action, self.low, self.high)

    def id(self):
        return f"optimistic_explorer_beta_{self.beta:.2f}"

    def __str__(self):
        return f"{self.name} (beta={self.beta})"


class CuriousExplorer(ExplorerStrategy):
    """
    Curious Explorer using Random Network Distillation (RND).
    Adds intrinsic reward based on prediction error of a fixed random network.
    Ref: Exploration by Random Network Distillation (Burda et al., 2018)
    """
    def __init__(self, agent, sub_cfg, low=-1, high=1, name: str = "curious-explorer"):
        super().__init__(agent.cfg.n_actions, low, high, name)
        self.beta = sub_cfg.beta # Intrinsic reward scale
        self.device = agent.actor.device

        state_dim = agent.cfg.input_dims[0]
        hidden_dim = sub_cfg.hidden_dim
        lr_pred = sub_cfg.lr_pred

        self.target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        ).to(self.device)
        for param in self.target.parameters():
            param.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        ).to(self.device)

        self.optimizer = T.optim.Adam(self.predictor.parameters(), lr=lr_pred)

        self.obs_buffer = []

    def choose_action(self, observation, agent=None, step=None, warmup=False):
        if warmup:
             return np.random.uniform(low=self.low, high=self.high, size=self.n_actions)

        return agent.choose_action_from_policy(observation)

    def choose_action_batch(self, observations, agent=None, step=None, warmup=False):
        if warmup:
             return np.random.uniform(low=self.low, high=self.high, size=(self.num_envs, self.n_actions))

        return agent.choose_action_from_policy_batch(observations)

    def get_intrinsic_reward(self, state, action, next_state):
        next_state_t = T.tensor(next_state, dtype=T.float32, device=self.device).unsqueeze(0)
        with T.no_grad():
            target_feat = self.target(next_state_t)
            pred_feat = self.predictor(next_state_t)

        error = T.sum((pred_feat - target_feat) ** 2, dim=-1)
        intrinsic_reward = self.beta * error.item()

        self.obs_buffer.append(next_state)

        return intrinsic_reward

    def end_episode(self):
        if not self.obs_buffer:
            return

        states = np.array(self.obs_buffer)
        states_t = T.tensor(states, dtype=T.float32, device=self.device)

        with T.no_grad():
            target_feat = self.target(states_t)

        pred_feat = self.predictor(states_t)
        loss = T.mean((pred_feat - target_feat) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.obs_buffer = []

    def reset(self):
        self.obs_buffer = []

    def id(self):
        return f"curious_explorer_beta_{self.beta:.2f}"

    def __str__(self):
        return f"{self.name} (beta={self.beta})"
