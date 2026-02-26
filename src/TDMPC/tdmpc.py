import numpy as np
import torch
import torch.nn as nn
import wandb
from copy import deepcopy
from omegaconf import OmegaConf
from pathlib import Path
import colorednoise
import os

from src.TDMPC import init
from src.TDMPC.action_hints import get_action_hints
import src.TDMPC.helper as h

CONFIG_FILE = "config.yaml"
MODEL_FILE = "model.pt"


class TOLD(nn.Module):
    """Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._encoder = h.mlp(
            cfg.obs_shape[0],
            max(cfg.num_enc_layers - 1, 1) * [cfg.enc_dim],
            cfg.latent_dim,
            act=h.SimNorm(cfg),
        )
        self._dynamics = h.mlp(
            cfg.latent_dim + cfg.action_dim,
            [cfg.mlp_dim] * 2,
            cfg.latent_dim,
            act=h.SimNorm(cfg),
        )
        self._reward = h.mlp(
            cfg.latent_dim + cfg.action_dim, 2 * [cfg.mlp_dim], max(cfg.num_bins, 1)
        )
        self._pi = h.mlp(cfg.latent_dim, 2 * [cfg.mlp_dim], cfg.action_dim)
        self._Q1 = h.mlp(
            cfg.latent_dim + cfg.action_dim,
            2 * [cfg.mlp_dim],
            max(cfg.num_bins, 1),
            dropout=cfg.dropout,
        )
        self._Q2 = h.mlp(
            cfg.latent_dim + cfg.action_dim,
            2 * [cfg.mlp_dim],
            max(cfg.num_bins, 1),
            dropout=cfg.dropout,
        )

        self.apply(init.weight_init)
        for m in [self._reward, self._Q1, self._Q2]:
            m[-1].weight.data.fill_(0)

    def track_q_grad(self, enable=True):
        """Utility function. Enables/disables gradient tracking of Q-networks."""
        for m in [self._Q1, self._Q2]:
            h.set_requires_grad(m, enable)

    def encode(self, obs):
        """Encodes an observation into its latent representation (h)."""
        return self._encoder(obs)

    def next(self, z, a):
        """Predicts next latent state (d)."""
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x)

    def pi(self, z, std=0):
        """Samples an action from the learned policy (pi)."""
        mu = torch.tanh(self._pi(z))
        if std > 0:
            std = torch.ones_like(mu) * std
            return h.TruncatedNormal(mu, std).sample(clip=0.3)
        return mu

    def reward_raw(self, z, a):
        """Predicts the reward distribution in log-space."""
        x = torch.cat([z, a], dim=-1)
        return self._reward(x)

    def reward(self, z, a):
        """Predicts the actual reward value by transforming the predicted distribution."""
        reward_raw = self.reward_raw(z, a)
        return h.two_hot_inv(reward_raw, self.cfg)

    def Q_raw(self, z, a):
        """Predicts the two Q-value distributions before applying the distributional inverse transform."""
        x = torch.cat([z, a], dim=-1)
        return self._Q1(x), self._Q2(x)

    def Q_min(self, z, a):
        """Predict minimum of the two Q-values."""
        Q_raw_1, Q_raw_2 = self.Q_raw(z, a)
        Q_1 = h.two_hot_inv(Q_raw_1, self.cfg)
        Q_2 = h.two_hot_inv(Q_raw_2, self.cfg)
        return torch.min(Q_1, Q_2)

    def get_non_policy_optimizer_config(self):
        """Return the parameters and learning rate for the optimizer of the non-policy part of the model."""
        return [
            {
                "params": self._encoder.parameters(),
                "lr": self.cfg.lr * self.cfg.get("enc_lr_scale", 1.0),
            },
            {"params": self._dynamics.parameters(), "lr": self.cfg.lr},
            {"params": self._reward.parameters(), "lr": self.cfg.lr},
            {"params": self._Q1.parameters(), "lr": self.cfg.lr},
            {"params": self._Q2.parameters(), "lr": self.cfg.lr},
        ]


class TDMPC:
    """Implementation of TD-MPC learning + inference."""

    def __init__(self, cfg, load_dir: str | Path = None):
        self.cfg = cfg
        if load_dir is not None:
            self.load_config(load_dir)
        self.device = torch.device("cuda")
        self.std = h.linear_schedule(self.cfg.std_schedule, 0)
        self.model = TOLD(self.cfg).cuda()
        self.model_target = deepcopy(self.model)
        self.optim = torch.optim.Adam(
            self.model.get_non_policy_optimizer_config(), lr=self.cfg.lr
        )
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr)

        self.planner_name = self.cfg.get("planner_name", "default")
        self.planning_noise_beta = self.cfg.get("planning_noise_beta", 0.25)
        self.fraction_elites_reused = self.cfg.get("fraction_elites_reused", 0.0)
        self.previous_elites = None
        self.shoot_bias = self.cfg.get("shoot_bias", 0.0)
        self.use_action_hints = self.cfg.get("use_action_hints", False)

        self.action_repeat = self.cfg.get("action_repeat", 1)
        self.previous_action = None
        self.same_action_counter = 0  # For action repeat

        self.model.eval()
        self.model_target.eval()
        if load_dir is not None:
            self.load(load_dir)

        self.static_batch = None

        self.select_plan_function()
        if self.cfg.compile and os.environ.get("LOCAL", "false").lower() == "false":
            self._plan = torch.compile(self._plan, mode="default")
            self._update = torch.compile(self._update, mode="reduce-overhead")

    def select_plan_function(self):
        match self.planner_name.lower():
            case "default":
                self._plan = self._default_plan
            case "icem":
                self._plan = self.iCEM
            case _:
                raise ValueError(f"Unknown planner name: {self.planner_name}")

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
            # First reward, then next state
            reward = self.model.reward(z, actions[t])
            z = self.model.next(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount
        G += discount * self.model.Q_min(z, self.model.pi(z, self.cfg.min_std))
        ## In TDMPCv2, they use average of 2/5 Q-values instead of min
        return G

    @torch.no_grad()
    def plan(self, obs: np.ndarray, eval_mode=False, step=None, t0=True):
        if not t0 and self.same_action_counter < self.action_repeat:
            self.same_action_counter += 1
            return self.previous_action

        if step < self.cfg.seed_steps and not eval_mode:
            self.same_action_counter = 1
            self.previous_action = torch.empty(
                self.cfg.action_dim, dtype=torch.float32, device=self.device
            ).uniform_(-1, 1)
            return self.previous_action

        if t0:
            self._prev_mean = torch.zeros(
                self.cfg.horizon, self.cfg.action_dim, device=self.device
            )
            self._prev_mean[:, -1] = self.shoot_bias
            self.previous_elites = None
        self.model.eval()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if self.use_action_hints:
            self.action_hints = torch.from_numpy(
                get_action_hints(obs.cpu().numpy(), self.cfg.horizon)
            ).cuda()

        planned_action, std = self._plan(obs)
        if not eval_mode:
            planned_action += std * torch.randn(self.cfg.action_dim, device=std.device)

        self.same_action_counter = 1
        self.previous_action = planned_action
        return planned_action

    @torch.no_grad()
    def iCEM(self, obs: torch.Tensor):
        """
        Plan next action using TD-MPC inference.
        obs: raw input observation
        """
        obs = obs.unsqueeze(0)
        # Sample policy trajectories
        horizon = self.cfg.horizon
        num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
        if num_pi_trajs > 0:
            pi_actions = torch.empty(
                horizon, num_pi_trajs, self.cfg.action_dim, device=self.device
            )
            z = self.model.encode(obs).repeat(num_pi_trajs, 1)
            for t in range(horizon):
                pi_actions[t] = self.model.pi(z, self.cfg.min_std)
                z = self.model.next(z, pi_actions[t])

        # Initialize state and parameters
        num_elites_reused = int(self.fraction_elites_reused * self.cfg.num_elites)
        z = self.model.encode(obs).repeat(
            self.cfg.num_samples + num_pi_trajs + num_elites_reused, 1
        )
        mean = torch.zeros(horizon, self.cfg.action_dim, device=self.device)
        std = 2 * torch.ones(horizon, self.cfg.action_dim, device=self.device)

        mean[:-1] = self._prev_mean[1:]
        previous_elites = (
            None if self.previous_elites is None else self.previous_elites[1:]
        )
        if previous_elites is not None:
            previous_elites = torch.cat(
                [
                    previous_elites,
                    torch.randn(  # TODO: maybe colored noise here as well?
                        1, num_elites_reused, self.cfg.action_dim, device=self.device
                    ),
                ],
                dim=0,
            )

        # Iterate CEM
        for i in range(self.cfg.iterations):
            actions = (
                torch.tensor(
                    colorednoise.powerlaw_psd_gaussian(
                        self.planning_noise_beta,
                        size=(self.cfg.num_samples, mean.shape[1], mean.shape[0]),
                    ).transpose([2, 0, 1])
                )
                .to(self.device)
                .type(torch.float32)
            )
            actions = torch.clamp(
                mean.unsqueeze(1) + std.unsqueeze(1) * actions,
                -1,
                1,
            )

            # Add mean in last iteration
            if i == self.cfg.iterations - 1:
                actions[:, -1] = mean

            # Reuse previous elites
            if previous_elites is not None and num_elites_reused > 0:
                actions = torch.cat([actions, previous_elites], dim=1)

            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)

            # Compute elite actions
            value = self.estimate_value(
                z[: actions.shape[1]], actions, horizon
            ).nan_to_num_(0)
            elite_idxs = torch.topk(
                value.squeeze(1), self.cfg.num_elites, dim=0
            ).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]
            previous_elites_indices = torch.topk(
                elite_value.squeeze(1), num_elites_reused, dim=0
            ).indices
            previous_elites = elite_actions[:, previous_elites_indices]

            # Update parameters (weighted update, not typical CEM)
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (
                score.sum(0) + 1e-9
            )
            _std = torch.sqrt(
                torch.sum(
                    score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2,
                    dim=1,
                )
                / (score.sum(0) + 1e-9)
            )
            _std = _std.clamp_(self.std, 2)
            mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std

        self._prev_mean = mean
        self.previous_elites = previous_elites

        score = score.squeeze(1)
        best_idx = score.argmax()
        return elite_actions[0, best_idx], _std[0]

    @torch.no_grad()
    def _default_plan(self, obs: torch.Tensor):
        """
        Plan next action using TD-MPC inference.
        obs: raw input observation
        """
        obs = obs.unsqueeze(0)
        # Sample policy trajectories
        horizon = self.cfg.horizon
        num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
        if num_pi_trajs > 0:
            pi_actions = torch.empty(
                horizon, num_pi_trajs, self.cfg.action_dim, device=self.device
            )
            z = self.model.encode(obs).repeat(num_pi_trajs, 1)
            for t in range(horizon):
                pi_actions[t] = self.model.pi(z, self.cfg.min_std)
                z = self.model.next(z, pi_actions[t])

        # Initialize state and parameters
        num_elites_reused = int(self.fraction_elites_reused * self.cfg.num_elites)
        z = self.model.encode(obs).repeat(
            self.cfg.num_samples + num_pi_trajs + num_elites_reused, 1
        )
        mean = torch.zeros(horizon, self.cfg.action_dim, device=self.device)
        mean[:, -1] = self.shoot_bias
        mean[:-1] = self._prev_mean[1:]
        std = 2 * torch.ones(horizon, self.cfg.action_dim, device=self.device)
        previous_elites = None

        # Iterate CEM
        for i in range(self.cfg.iterations):
            actions = torch.clamp(
                mean.unsqueeze(1)
                + std.unsqueeze(1)
                * torch.randn(
                    horizon,
                    self.cfg.num_samples,
                    self.cfg.action_dim,
                    device=std.device,
                ),
                -1,
                1,
            )
            if i == 0 and self.use_action_hints:
                actions[:, : self.action_hints.shape[1], :] = self.action_hints
            if self.previous_elites is not None and num_elites_reused > 0:
                actions = torch.cat([actions, previous_elites], dim=1)

            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)

            # Compute elite actions
            value = self.estimate_value(
                z[: actions.shape[1]], actions, horizon
            ).nan_to_num_(0)
            elite_idxs = torch.topk(
                value.squeeze(1), self.cfg.num_elites, dim=0
            ).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]
            previous_elites_indices = torch.topk(
                elite_value.squeeze(1), num_elites_reused, dim=0
            ).indices
            previous_elites = elite_actions[:, previous_elites_indices]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (
                score.sum(0) + 1e-9
            )
            _std = torch.sqrt(
                torch.sum(
                    score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2,
                    dim=1,
                )
                / (score.sum(0) + 1e-9)
            )
            _std = _std.clamp_(self.std, 2)
            mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std

        # Outputs
        score = score.squeeze(1)
        idx = torch.multinomial(score, 1)
        actions = elite_actions.index_select(1, idx).squeeze(1)
        self._prev_mean = mean
        mean, std = actions[0], _std[0]
        return mean, std

    def update_pi(self, zs):
        """Update policy using a sequence of latent states."""
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)

        # Loss is a weighted sum of Q-values
        pi_loss = 0
        for t, z in enumerate(zs):
            a = self.model.pi(z, self.cfg.min_std)
            Q = self.model.Q_min(z, a)
            pi_loss += -Q.mean() * (self.cfg.rho**t)

        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model._pi.parameters(),
            self.cfg.grad_clip_norm,
            error_if_nonfinite=False,
        )
        self.pi_optim.step()
        self.model.track_q_grad(True)
        return pi_loss

    @torch.no_grad()
    def _td_target(self, next_obs, reward, done):
        """Compute the TD-target from a reward and the observation at the following time step."""
        # TODO: Should model_target be used for next_z computation?
        next_z = self.model.encode(next_obs)
        td_target = reward + self.cfg.discount * (1 - done) * self.model_target.Q_min(
            next_z, self.model.pi(next_z, self.cfg.min_std)
        )
        return td_target

    def update(self, replay_buffer, step):
        self.std = h.linear_schedule(self.cfg.std_schedule, step)
        batch = replay_buffer.sample()

        if self.static_batch is None:
            self.static_batch = [
                t.detach().clone() if isinstance(t, torch.Tensor) else t for t in batch
            ]
        for i in range(len(self.static_batch)):
            if isinstance(batch[i], torch.Tensor):
                self.static_batch[i].copy_(batch[i])
            else:
                self.static_batch[i] = batch[i]

        torch.compiler.cudagraph_mark_step_begin()
        out_tensors = self._update(self.static_batch)

        # Optimizer step outside of compiled function
        self.optim.step()

        # Update policy + target network
        pi_loss = self.update_pi(out_tensors["zs"])
        if step % self.cfg.update_freq == 0:
            h.ema(self.model, self.model_target, self.cfg.tau)

        idxs = self.static_batch[-2]
        replay_buffer.update_priorities(idxs, out_tensors["priority_loss"])

        out_tensors.pop("priority_loss")
        out_tensors.pop("zs")
        out_tensors["pi_loss"] = pi_loss

        losses = {k: float(v.item()) for k, v in out_tensors.items()}
        return losses

    def _update(self, batch):
        """Main update function. Corresponds to one iteration of the TOLD model learning."""
        obs, next_obses, action, reward, done, _, weights = batch
        self.optim.zero_grad(set_to_none=True)
        self.model.train()

        # Representation
        z = self.model.encode(obs)
        zs = [z.detach()]

        mask = torch.ones_like(done[0])
        consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
        for t in range(self.cfg.horizon):
            # Predictions
            Q1, Q2 = self.model.Q_raw(z, action[t])
            reward_pred = self.model.reward_raw(z, action[t])
            z = self.model.next(z, action[t])
            with torch.no_grad():
                next_obs = next_obses[t]
                next_z = self.model_target.encode(next_obs)
                td_target = self._td_target(next_obs, reward[t], done[t])
            zs.append(z.detach())

            # Losses
            rho = self.cfg.rho**t
            consistency_loss += rho * torch.mean(
                h.mse(z, next_z, mask), dim=1, keepdim=True
            )
            reward_loss += rho * h.soft_ce(reward_pred, reward[t], mask, self.cfg)
            value_loss += rho * (
                h.soft_ce(Q1, td_target, mask, self.cfg)
                + h.soft_ce(Q2, td_target, mask, self.cfg)
            )
            priority_loss += rho * (
                h.l1(h.two_hot_inv(Q1, self.cfg), td_target, mask)
                + h.l1(h.two_hot_inv(Q2, self.cfg), td_target, mask)
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

        return {
            "consistency_loss": consistency_loss.mean(),
            "reward_loss": reward_loss.mean(),
            "value_loss": value_loss.mean(),
            "total_loss": total_loss.mean(),
            "weighted_loss": weighted_loss.mean(),
            "grad_norm": grad_norm,
            "priority_loss": priority_loss.clamp(max=1e4).detach(),
            "zs": zs,
        }
