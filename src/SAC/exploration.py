import numpy as np
import torch as T
import torch.nn as nn
import torch.autograd

# Exploration Strategies for Continuous Action Spaces

class ExplorerStrategy:
    """
    Base class for exploration strategies.
    """
    def choose_action(self, x):
        raise NotImplementedError

    def reset(self):
        pass

    def start_episode(self):
        pass

    def end_episode(self):
        pass

class RandomExplorer(ExplorerStrategy):
    """
    Random (Uniform) exploration strategy for continuous action spaces.
    """
    def __init__(self, n_actions: int, low=-1, high=1, name: str = "uniform-explorer"):
        self.n_actions = n_actions
        self.name = name
        self.low = low
        self.high = high

    def choose_action(self, x):
        return np.random.uniform(low=self.low, high=self.high, size=self.n_actions)

    def __str__(self):
        return self.name + f" ({self.n_actions} actions)"

    def id(self):
        return "uniform_explorer"

class GaussianExplorer(ExplorerStrategy):
    """
    Gaussian exploration strategy for continuous action spaces.
    """
    def __init__(self, n_actions: int, mu=0.0, sigma=0.8, low=-1, high=1, name: str = "gaussian-explorer"):
        self.n_actions = n_actions
        self.mu = mu
        self.sigma = sigma
        self.name = name
        self.low = low
        self.high = high

    def choose_action(self, x):
        out = np.random.normal(loc=self.mu, scale=self.sigma, size=self.n_actions)
        return np.clip(out, self.low, self.high)

    def __str__(self):
        return self.name + f" ({self.n_actions} actions, mu={self.mu}, sigma={self.sigma})"

    def id(self):
        if self.mu != 0.0:
            return f"gaussian_explorer_mu_{self.mu:.2f}_sigma_{self.sigma:.2f}"
        return f"gaussian_explorer_sigma_{self.sigma:.2f}"

class OrnsteinUhlenbeckExplorer(ExplorerStrategy):
    """
    Ornstein-Uhlenbeck exploration strategy for continuous action spaces.
    """
    def __init__(self, n_actions: int, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2, x0=None, max_episodes=int(1e5 // 200), low=-1, high=1, name: str = "ou-explorer"):
        self.n_actions = n_actions
        self.mu = mu
        self.theta = theta
        self.init_sigma = sigma
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.name = name
        self.low = low
        self.high = high
        self.max_episodes = max_episodes
        self.reset()

    def reset(self):
        if self.x0 is not None:
            self.x_prev = self.x0
        else:
            self.x_prev = np.zeros(self.n_actions)

    def anneal_sigma(self):
        self.sigma = self.sigma * min((1 - 1 / self.max_episodes), 0.99)

    def choose_action(self, x):
        x = (self.x_prev +
             self.theta * (self.mu - self.x_prev) * self.dt +
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.n_actions))
        x = np.clip(x, self.low, self.high)
        self.x_prev = x
        return x

    def end_episode(self):
        self.anneal_sigma()

    def __str__(self):
        return (self.name + f" ({self.n_actions} actions, mu={self.mu}, "
                f"theta={self.theta}, sigma0={self.init_sigma}, dt={self.dt})")

    def id(self):
        return f"ou_explorer_theta_{self.theta:.2f}_sigma_{self.init_sigma:.2f}"

class OptimisticExplorer(ExplorerStrategy):
    """
    Optimistic exploration strategy for continuous action spaces.
    Ref: Better Exploration with Optimistic Actor-Critic
         Ciosek, et al., 2020
    """
    def __init__(self, agent, sub_cfg, low=-1, high=1, name: str = "optimistic-explorer"):
        self.agent = agent
        self.n_actions = agent.cfg.n_actions
        self.beta = sub_cfg.beta
        self.low = low
        self.high = high
        self.name = name

    def choose_action(self, state):
        # mux, stdx from agent's Actor
        # compute q_avg = (critic1 + critic2) / 2 at (state, mux)
        # compute gradient of q_avg wrt mux
        # mu_opt = mux + beta * stdx * grad_q_avg_wrt_mux
        # action = sample from N(mu_opt, stdx)
        # tanh and clip action to [low, high]
        state_tensor = T.tensor(state, dtype=T.float32).unsqueeze(0)
        with T.no_grad():
            mux, stdx = self.agent.actor.forward(state_tensor)
        mux.requires_grad_(True)
        q1_opt = self.agent.critic_1.forward(state_tensor, mux)
        q2_opt = self.agent.critic_2.forward(state_tensor, mux)
        q_avg_opt = (q1_opt + q2_opt) / 2.0
        q_avg_opt.backward()
        grad_q_avg_wrt_mux = mux.grad
        mu_opt = mux + self.beta * stdx * grad_q_avg_wrt_mux
        mu_opt = mu_opt.detach()
        action = T.normal(mu_opt, stdx)
        action = T.tanh(action)
        action = action.squeeze(0).cpu().detach().numpy()
        action = np.clip(action, self.low, self.high)
        return action

    def __str__(self):
        return self.name + f" (beta={self.beta})"

    def id(self):
        return f"optimistic_explorer_beta_{self.beta:.2f}"

class CuriousExplorer(ExplorerStrategy):
    """
    Curious exploration strategy for continuous action spaces.
    Ref: Exploration by Random Network Distillation
         Burda, et al., 2018
    """
    def __init__(self, agent, sub_cfg, low=-1, high=1, name: str = "curious-explorer"):
        self.cfg = agent.cfg
        state_dim = agent.cfg.input_dims[0]
        n_actions = agent.cfg.n_actions
        self.n_actions = n_actions
        self.low = low
        self.high = high
        self.name = name
        self.beta = sub_cfg.beta

        self.obs_list = []

        # Curiosity Networks
        self.target = nn.Sequential(
            nn.Linear(state_dim, sub_cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(sub_cfg.hidden_dim, sub_cfg.hidden_dim),
        )
        for param in self.target.parameters():
            param.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(state_dim, sub_cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(sub_cfg.hidden_dim, sub_cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(sub_cfg.hidden_dim, sub_cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(sub_cfg.hidden_dim, sub_cfg.hidden_dim),
        )

        self.optimizer = T.optim.Adam(self.predictor.parameters(), lr=sub_cfg.lr_pred)
        self.agent = agent
        self.target.to(agent.actor.device)
        self.predictor.to(agent.actor.device)

    def compute_novelty(self, state):
        with T.no_grad():
            target_features = self.target(state)
        predicted_features = self.predictor(state)
        novelty = T.mean((predicted_features - target_features) ** 2, dim=-1)
        return novelty.item()

    def choose_action(self, state):
        # mux, stdx from agent's actor
        # compute_novelty for state
        # update std = stdx * (1 + novelty * beta)
        # action = sample from N(mux, std)
        # tanh and clip action to [low, high]
        self.obs_list.append(state)
        state_tensor = T.tensor(state, dtype=T.float32).unsqueeze(0)
        with T.no_grad():
            mux, stdx = self.agent.actor.forward(state_tensor)
        novelty = self.compute_novelty(state_tensor)
        std_modified = stdx * (1 + novelty * self.beta)
        action = T.normal(mux, std_modified)
        action = T.tanh(action)
        action = action.squeeze(0).cpu().detach().numpy()
        action = np.clip(action, self.low, self.high)
        return action

    def end_episode(self):
        # Update predictor network using collected states
        if len(self.obs_list) == 0:
            return
        states = np.array(self.obs_list)
        states_tensor = T.tensor(states, dtype=T.float32, device=self.agent.actor.device)
        with T.no_grad():
            target_features = self.target(states_tensor)
        predicted_features = self.predictor(states_tensor)
        loss = T.mean((predicted_features - target_features) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.obs_list = []

    def reset(self):
        self.obs_list = []

    def __str__(self):
        return self.name + f" (beta={self.beta})"

    def id(self):
        return f"curious_explorer_beta_{self.beta:.2f}"


if __name__ == "__main__":
    import hockey.hockey_env as h_env
    from helper import HeatmapTracker, load_agent_from_config
    from agent import Agent
    from omegaconf import OmegaConf

    env = h_env.HockeyEnv()
    agent = load_agent_from_config('reward-v0-per-7_3-bot-pool', env)
    heatmap_logger = HeatmapTracker()
    n_actions = env.action_space.shape[0] // 2
    n_episodes = 100
    explorers = [
        # random-explorer
        RandomExplorer(n_actions),
        # gaussian-explorer: sigma
        GaussianExplorer(n_actions, sigma=0.2),
        GaussianExplorer(n_actions, sigma=0.5),
        GaussianExplorer(n_actions, sigma=0.8),
        # ou-explorer: sigma
        OrnsteinUhlenbeckExplorer(n_actions, sigma=0.2, max_episodes=n_episodes),
        OrnsteinUhlenbeckExplorer(n_actions, sigma=0.5, max_episodes=n_episodes),
        OrnsteinUhlenbeckExplorer(n_actions, sigma=0.8, max_episodes=n_episodes),
        # ou-explorer: theta
        OrnsteinUhlenbeckExplorer(n_actions, theta=0.05, max_episodes=n_episodes),
        OrnsteinUhlenbeckExplorer(n_actions, theta=0.15, max_episodes=n_episodes),
        OrnsteinUhlenbeckExplorer(n_actions, theta=0.3, max_episodes=n_episodes),
        # optimistic-explorer: beta
        OptimisticExplorer(agent, beta=0.1),
        OptimisticExplorer(agent, beta=0.5),
        OptimisticExplorer(agent, beta=1),
        # curious-explorer: beta
        CuriousExplorer(agent, beta=0.1),
        CuriousExplorer(agent, beta=0.5),
        CuriousExplorer(agent, beta=1.0),
    ]
    random_explorer = RandomExplorer(n_actions)
    for explorer in explorers:
        heatmap_logger.reset()
        for i in range(n_episodes):
            all_frames = []
            obs, _ = env.reset()
            while True:
                heatmap_logger.record_step(obs)
                agent_action = explorer.choose_action(obs)
                opponent_action = random_explorer.choose_action()
                obs, reward, done, truncated, info = env.step(np.hstack([agent_action, opponent_action]))
                all_frames.append(obs)
                if done or truncated:
                    obs, _ = env.reset()
                    break
            explorer.end_episode()
        entropy = heatmap_logger.compute_entropy()
        heatmap_logger.save_heatmap(filename=f'heatmap_{explorer.id()}.png')
        print(str( explorer ) + f" - Entropy: {entropy:.4f}")
