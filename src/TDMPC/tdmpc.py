import numpy as np
import torch
import torch.nn as nn
import wandb
from copy import deepcopy
from omegaconf import OmegaConf
from pathlib import Path

import src.TDMPC.helper as h

CONFIG_FILE = "config.yaml"
MODEL_FILE = "model.pt"


class TOLD(nn.Module):
    """Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._encoder = h.enc(cfg)
        self._dynamics = h.mlp(
            cfg.latent_dim + cfg.action_dim, cfg.mlp_dim, cfg.latent_dim
        )
        self._reward = h.mlp(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim, 1)
        self._pi = h.mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
        self._Q1, self._Q2 = h.q(cfg), h.q(cfg)
        self.apply(h.orthogonal_init)
        for m in [self._reward, self._Q1, self._Q2]:
            m[-1].weight.data.fill_(0)
            m[-1].bias.data.fill_(0)

    def track_q_grad(self, enable=True):
        """Utility function. Enables/disables gradient tracking of Q-networks."""
        for m in [self._Q1, self._Q2]:
            h.set_requires_grad(m, enable)

    def h(self, obs):
        """Encodes an observation into its latent representation (h)."""
        return self._encoder(obs)

    def next(self, z, a):
        """Predicts next latent state (d) and single-step reward (R)."""
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x), self._reward(x)

    def pi(self, z, std=0):
        """Samples an action from the learned policy (pi)."""
        mu = torch.tanh(self._pi(z))
        if std > 0:
            std = torch.ones_like(mu) * std
            return h.TruncatedNormal(mu, std).sample(clip=0.3)
        return mu

    def Q(self, z, a):
        """Predict state-action value (Q)."""
        x = torch.cat([z, a], dim=-1)
        return self._Q1(x), self._Q2(x)


class TDMPC:
    """Implementation of TD-MPC learning + inference."""

    def __init__(self, cfg, load_dir: str | Path = None):
        self.cfg = cfg
        if load_dir is not None:
            self.load_config(load_dir)
        self.device = torch.device("cuda")
        # TODO: move to config
        self.std = h.linear_schedule(self.cfg.std_schedule, 0)
        self.model = TOLD(self.cfg).cuda()
        self.model_target = deepcopy(self.model)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr)
        self.model.eval()
        self.model_target.eval()
        if load_dir is not None:
            self.load(load_dir)

    def state_dict(self):
        """Retrieve state dict of TOLD model, including slow-moving target network."""
        return {
            "model": self.model.state_dict(),
            "model_target": self.model_target.state_dict(),
        }

    def save(self, path: str | Path):
        """Save state dict of TOLD model to filepath."""
        path = Path(path)
        torch.save(self.state_dict(), path / MODEL_FILE)
        with open(path / CONFIG_FILE, "w") as f:
            OmegaConf.save(self.cfg, f)

    def save_to_wandb(self, wandb_run: wandb.Run, step: int):
        """Save model and config to wandb artifact."""
        save_dir = Path(wandb_run.dir) / f"checkpoint_step_{step}"
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save(save_dir)

        wandb.save(str(save_dir / MODEL_FILE))
        wandb.save(str(save_dir / CONFIG_FILE))

    def load_config(self, path: str | Path):
        """Load config from filepath."""
        path = Path(path)
        with open(path / CONFIG_FILE, "r") as f:
            self.cfg = OmegaConf.load(f)

    def load_state_dict(self, state_dict: dict):
        """Load a saved state dict."""
        self.model.load_state_dict(state_dict["model"])
        self.model_target.load_state_dict(state_dict["model_target"])

    def load(self, path: str | Path):
        """Load a saved state dict and  current agent."""
        path = Path(path)
        self.load_config(path)
        d = torch.load(path / MODEL_FILE)
        self.load_state_dict(d)

    @torch.no_grad()
    def estimate_value(self, z, actions, horizon):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(horizon):
            z, reward = self.model.next(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount
        G += discount * torch.min(*self.model.Q(z, self.model.pi(z, self.cfg.min_std)))
        return G

    @torch.no_grad()
    def plan(self, obs, eval_mode=False, step=None, t0=True):
        """
        Plan next action using TD-MPC inference.
        obs: raw input observation. Shape (batch_size, obs_dim).
        eval_mode: uniform sampling and action noise is disabled during evaluation.
        step: current time step. determines e.g. planning horizon.
        t0: whether current step is the first step of an episode.
        """
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        batch_size = obs.shape[0]

        if step < self.cfg.seed_steps and not eval_mode:
            return torch.empty(
                batch_size, self.cfg.action_dim, dtype=torch.float32, device=self.device
            ).uniform_(-1, 1)

        z_batch = self.model.h(obs)

        horizon = int(
            min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step))
        )
        num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)

        if num_pi_trajs > 0:
            z_pi = (
                z_batch.unsqueeze(0)
                .repeat(num_pi_trajs, 1, 1)
                .reshape(num_pi_trajs * batch_size, -1)
            )
            pi_actions = torch.empty(
                horizon,
                num_pi_trajs * batch_size,
                self.cfg.action_dim,
                device=self.device,
            )

            for t in range(horizon):
                pi_actions[t] = self.model.pi(z_pi, self.cfg.min_std)
                z_pi, _ = self.model.next(z_pi, pi_actions[t])

            # Reshape back
            pi_actions = pi_actions.view(
                horizon, num_pi_trajs, batch_size, self.cfg.action_dim
            )

        mean = torch.zeros(horizon, batch_size, self.cfg.action_dim, device=self.device)
        std = 2 * torch.ones(
            horizon, batch_size, self.cfg.action_dim, device=self.device
        )

        if (
            not t0
            and hasattr(self, "_prev_mean")
            and self._prev_mean.shape[1] == batch_size
        ):
            prev_mean_shifted = self._prev_mean[1:]
            valid_len = min(prev_mean_shifted.shape[0], mean.shape[0] - 1)
            if valid_len > 0:
                mean[:valid_len] = prev_mean_shifted[:valid_len]

        for i in range(self.cfg.iterations):
            noise = torch.randn(
                horizon,
                self.cfg.num_samples,
                batch_size,
                self.cfg.action_dim,
                device=self.device,
            )

            actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * noise, -1, 1)

            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)

            P = actions.shape[1]
            actions_flat = actions.reshape(horizon, P * batch_size, self.cfg.action_dim)

            z_flat = z_batch.unsqueeze(0).repeat(P, 1, 1).reshape(P * batch_size, -1)

            value = self.estimate_value(z_flat, actions_flat, horizon).nan_to_num_(0)
            value = value.view(P, batch_size)

            elite_idxs = torch.topk(value, self.cfg.num_elites, dim=0).indices
            elite_value = torch.gather(value, 0, elite_idxs)

            elite_idxs_expanded = elite_idxs.view(
                1, self.cfg.num_elites, batch_size, 1
            ).expand(horizon, -1, -1, self.cfg.action_dim)
            elite_actions = torch.gather(actions, 1, elite_idxs_expanded)

            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score /= score.sum(0)

            _mean = torch.sum(
                score.view(1, self.cfg.num_elites, batch_size, 1) * elite_actions, dim=1
            ) / (score.sum(0).view(1, batch_size, 1) + 1e-9)

            _std = torch.sqrt(
                torch.sum(
                    score.view(1, self.cfg.num_elites, batch_size, 1)
                    * (elite_actions - _mean.unsqueeze(1)) ** 2,
                    dim=1,
                )
                / (score.sum(0).view(1, batch_size, 1) + 1e-9)
            )
            _std = _std.clamp_(self.std, 2)

            mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std

        score = score / score.sum(0, keepdim=True)
        sampled_idxs = torch.multinomial(score.T, 1).squeeze(1)
        sampled_idxs_expanded = sampled_idxs.view(1, 1, batch_size, 1).expand(
            horizon, -1, -1, self.cfg.action_dim
        )
        chosen_actions = torch.gather(elite_actions, 1, sampled_idxs_expanded).squeeze(
            1
        )

        self._prev_mean = mean
        a = chosen_actions[0]

        if not eval_mode:
            a += _std[0] * torch.randn(
                batch_size, self.cfg.action_dim, device=self.device
            )

        return a

    def update_pi(self, zs):
        """Update policy using a sequence of latent states."""
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)

        # Loss is a weighted sum of Q-values
        pi_loss = 0
        for t, z in enumerate(zs):
            a = self.model.pi(z, self.cfg.min_std)
            Q = torch.min(*self.model.Q(z, a))
            pi_loss += -Q.mean() * (self.cfg.rho**t)

        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model._pi.parameters(),
            self.cfg.grad_clip_norm,
            error_if_nonfinite=False,
        )
        self.pi_optim.step()
        self.model.track_q_grad(True)
        return pi_loss.item()

    @torch.no_grad()
    def _td_target(self, next_obs, reward, done):
        """Compute the TD-target from a reward and the observation at the following time step."""
        next_z = self.model.h(next_obs)
        td_target = reward + self.cfg.discount * (1 - done) * torch.min(
            *self.model_target.Q(next_z, self.model.pi(next_z, self.cfg.min_std))
        )
        return td_target

    def update(self, replay_buffer, step):
        """Main update function. Corresponds to one iteration of the TOLD model learning."""
        obs, next_obses, action, reward, done, idxs, weights = replay_buffer.sample()
        self.optim.zero_grad(set_to_none=True)
        self.std = h.linear_schedule(self.cfg.std_schedule, step)
        self.model.train()

        # Representation
        z = self.model.h(obs)
        zs = [z.detach()]

        mask = torch.ones_like(done[0])
        consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
        for t in range(self.cfg.horizon):
            # Predictions
            Q1, Q2 = self.model.Q(z, action[t])
            z, reward_pred = self.model.next(z, action[t])
            with torch.no_grad():
                next_obs = next_obses[t]
                next_z = self.model_target.h(next_obs)
                td_target = self._td_target(next_obs, reward[t], done[t])
            zs.append(z.detach())

            # Losses
            rho = self.cfg.rho**t
            consistency_loss += rho * torch.mean(
                h.mse(z, next_z, mask), dim=1, keepdim=True
            )
            reward_loss += rho * h.mse(reward_pred, reward[t], mask)
            value_loss += rho * (
                h.mse(Q1, td_target, mask) + h.mse(Q2, td_target, mask)
            )
            priority_loss += rho * (
                h.l1(Q1, td_target, mask) + h.l1(Q2, td_target, mask)
            )
            mask = mask * (1 - done[t])

        # Optimize model
        total_loss = (
            self.cfg.consistency_coef * consistency_loss.clamp(max=1e4)
            + self.cfg.reward_coef * reward_loss.clamp(max=1e4)
            + self.cfg.value_coef * value_loss.clamp(max=1e4)
        )
        weighted_loss = (total_loss.squeeze(1) * weights).mean()
        weighted_loss.register_hook(lambda grad: grad * (1 / self.cfg.horizon))
        weighted_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False
        )
        self.optim.step()
        replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())

        # Update policy + target network
        pi_loss = self.update_pi(zs)
        if (
            step % self.cfg.update_freq == 0
        ):  # TODO: investigate this, but works okay for now
            h.ema(self.model, self.model_target, self.cfg.tau)

        self.model.eval()
        return {
            "consistency_loss": float(consistency_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "value_loss": float(value_loss.mean().item()),
            "pi_loss": pi_loss,
            "total_loss": float(total_loss.mean().item()),
            "weighted_loss": float(weighted_loss.mean().item()),
            "grad_norm": float(grad_norm),
        }
