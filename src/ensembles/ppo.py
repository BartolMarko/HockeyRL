# Adapted from CleanRL: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from omegaconf import OmegaConf


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, num_actions: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, num_actions), std=0.01),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        return probs.sample()

    def get_action_and_value(
        self, x, action=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns action, logprob of the action, entropy, and value estimates for the given state.
        If action is not provided, it will be sampled from the policy.
        """
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class PPO:
    def __init__(
        self,
        cfg: OmegaConf,
        observation_dim: int,
        num_different_actions: int,
    ):
        self.cfg = cfg
        self.device = cfg.device
        self.observation_dim = observation_dim
        self.actor_critic = ActorCritic(observation_dim, num_different_actions).to(
            self.device
        )
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=cfg.learning_rate, eps=1e-5
        )

        self._reset_episode_storage()
        self._reset_batch_storage()

    def act(self, obs: np.ndarray) -> int:
        # TODO: maybe with torch.no_grad()
        obs = torch.Tensor(obs).to(self.device)
        action = self.actor_critic.get_action(obs)
        return action.cpu().numpy().item()

    def add_to_storage(
        self,
        obs: np.ndarray,
        action: int,
        logprob: float,
        value: float,
        done: bool,
    ):
        if not done:
            self.episode_obs = torch.cat(
                [self.episode_obs, torch.Tensor(obs).to(self.device).unsqueeze(0)]
            )
        self.episode_logprobs = torch.cat(
            [self.episode_logprobs, torch.Tensor([logprob]).to(self.device)]
        )
        self.episode_actions = torch.cat(
            [self.episode_actions, torch.Tensor([action]).to(self.device).unsqueeze(1)]
        )
        self.episode_values = torch.cat(
            [self.episode_values, torch.Tensor([value]).to(self.device)]
        )
        if done:
            returns, advantages = self._calculate_episode_returns_and_advantages()
            self.b_obs = torch.cat([self.b_obs, self.episode_obs])
            self.b_logprobs = torch.cat([self.b_logprobs, self.episode_logprobs])
            self.b_actions = torch.cat([self.b_actions, self.episode_actions])
            self.b_advantages = torch.cat([self.b_advantages, advantages])
            self.b_returns = torch.cat([self.b_returns, returns])
            self.b_values = torch.cat([self.b_values, self.episode_values])
            self._reset_episode_storage()

    def update(self, global_step: int):
        if self.cfg.anneal_lr:
            self._anneal_learning_rate(global_step)

        b_inds = np.arange(len(self.b_obs))
        minibatch_size = len(self.b_obs) // self.cfg.num_minibatches
        for epoch in range(self.cfg.update_epochs):
            np.random.shuffle(b_inds)
            for minibatch_idx in range(0, self.cfg.num_minibatches):
                start = minibatch_idx * minibatch_size
                end = (
                    start + minibatch_size
                    if minibatch_idx < self.cfg.num_minibatches - 1
                    else len(self.b_obs)
                )

                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = (
                    self.actor_critic.get_action_and_value(
                        self.b_obs[mb_inds], self.b_actions.long()[mb_inds]
                    )
                )
                logratio = newlogprob - self.b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = self.b_advantages[mb_inds]
                if self.cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - self.b_returns[mb_inds]) ** 2
                    v_clipped = self.b_values[mb_inds] + torch.clamp(
                        newvalue - self.b_values[mb_inds],
                        -self.cfg.clip_coef,
                        self.cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - self.b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - self.b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - self.cfg.ent_coef * entropy_loss
                    + v_loss * self.cfg.vf_coef
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

    def _anneal_learning_rate(self, global_step: int):
        frac = 1.0 - global_step / self.cfg.num_iterations
        lrnow = self.cfg.end_learning_rate + frac * (
            self.cfg.learning_rate - self.cfg.end_learning_rate
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lrnow

    def _calculate_episode_returns_and_advantages(self):
        """Calculates returns and advantages for the finished episode."""
        advantages = torch.zeros_like(self.episode_logprobs).to(self.device)

        lastgaelam, next_value = 0.0, 0.0
        # Automatically captures done=True at the end of the episode
        for t in reversed(range(len(self.episode_logprobs))):
            delta = (
                self.episode_rewards[t]
                + self.cfg.gamma * next_value
                - self.episode_values[t]
            )
            advantages[t] = delta + self.cfg.gamma * self.cfg.gae_lambda * lastgaelam
            next_value, lastgaelam = self.episode_values[t], advantages[t]

        returns = advantages + self.episode_values
        return returns, advantages

    def _reset_episode_storage(self):
        self.episode_obs = torch.zeros((0, self.observation_dim)).to(self.device)
        self.episode_logprobs = torch.zeros((0,)).to(self.device)
        self.episode_actions = torch.zeros((0, 1)).to(self.device)
        self.episode_values = torch.zeros((0,)).to(self.device)

    def _reset_batch_storage(self):
        self.b_obs = torch.zeros((0, self.observation_dim)).to(self.device)
        self.b_logprobs = torch.zeros((0,)).to(self.device)
        self.b_actions = torch.zeros((0, 1)).to(self.device)
        self.b_advantages = torch.zeros((0,)).to(self.device)
        self.b_returns = torch.zeros((0,)).to(self.device)
        self.b_values = torch.zeros((0,)).to(self.device)
