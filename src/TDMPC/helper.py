import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

from src.episode import Episode


__REDUCE__ = lambda b: "mean" if b else "none"  # noqa


def symlog(x):
    """
    Symmetric logarithmic function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
    """
    Symmetric exponential function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x, cfg):
    """Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
    if cfg.num_bins == 0:
        return x
    elif cfg.num_bins == 1:
        return symlog(x)
    x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax).squeeze(1)
    bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size)
    bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx).unsqueeze(-1)
    soft_two_hot = torch.zeros(x.shape[0], cfg.num_bins, device=x.device, dtype=x.dtype)
    bin_idx = bin_idx.long()
    soft_two_hot = soft_two_hot.scatter(1, bin_idx.unsqueeze(1), 1 - bin_offset)
    soft_two_hot = soft_two_hot.scatter(
        1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset
    )
    return soft_two_hot


def two_hot_inv(x, cfg):
    """Converts a batch of soft two-hot encoded vectors to scalars."""
    if cfg.num_bins == 0:
        return x
    elif cfg.num_bins == 1:
        return symexp(x)
    dreg_bins = torch.linspace(
        cfg.vmin, cfg.vmax, cfg.num_bins, device=x.device, dtype=x.dtype
    )
    x = F.softmax(x, dim=-1)
    x = torch.sum(x * dreg_bins, dim=-1, keepdim=True)
    return symexp(x)


def soft_ce(pred, target, mask, cfg):
    """Computes the cross entropy loss between predictions and soft targets."""
    pred = F.log_softmax(pred, dim=-1)
    target = two_hot(target, cfg)
    return -(target * pred * mask).sum(-1, keepdim=True)


def l1(pred, target, mask=None, reduce=False):
    """Computes the L1-loss between predictions and targets."""
    if mask is not None:
        pred = pred * mask
        target = target * mask
    return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))


def mse(pred, target, mask=None, reduce=False):
    """Computes the MSE loss between predictions and targets."""
    if mask is not None:
        pred = pred * mask
        target = target * mask
    return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))


def ema(m, m_target, tau):
    """Update slow-moving average of online network (target network) at rate tau."""
    with torch.no_grad():
        for p, p_target in zip(m.parameters(), m_target.parameters()):
            p_target.data.lerp_(p.data, tau)


def set_requires_grad(net, value):
    """Enable/disable gradients for a given (sub)network."""
    for param in net.parameters():
        param.requires_grad_(value)


class TruncatedNormal(pyd.Normal):
    """Utility class implementing the truncated normal distribution."""

    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.simnorm_dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)


class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, activation, and optionally dropout.
    """

    def __init__(self, *args, dropout=0.0, act=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        if act is None:
            act = nn.Mish(inplace=False)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.0):
    """
    Basic building block of TD-MPC2.
    MLP with LayerNorm, Mish activations, and optionally dropout.
    Last layer is linear if no activation is specified, else uses the given activation.
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0)))

    mlp.append(
        NormedLinear(dims[-2], dims[-1], act=act)
        if act
        else nn.Linear(dims[-2], dims[-1])
    )
    return nn.Sequential(*mlp)


class CircularIndices:
    def __init__(self, indices, size):
        self.indices = indices % size
        self.size = size

    def __add__(self, n: int):
        self.add(n)
        return self

    def add(self, n: int):
        self.indices = (self.indices + n) % self.size

    def get(self):
        return self.indices

    def pop_last(self):
        last_idx = self.indices[-1]
        self.indices = self.indices[:-1]
        return last_idx


class ReplayBuffer:
    """
    Storage and sampling functionality for training TD-MPC / TOLD.
    The replay buffer is stored in GPU memory when training from state.
    Uses prioritized experience replay by default."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.capacity = min(cfg.train_steps, cfg.max_buffer_size)
        dtype = torch.float32
        obs_shape = cfg.obs_shape
        self._obs = torch.empty(
            (self.capacity + 1, *obs_shape), dtype=dtype, device=self.device
        )
        self._action = torch.empty(
            (self.capacity, cfg.action_dim), dtype=torch.float32, device=self.device
        )
        self._opponent_action = torch.empty(
            (self.capacity, cfg.action_dim), dtype=torch.float32, device=self.device
        )
        self._done = torch.empty(
            (self.capacity,), dtype=torch.float32, device=self.device
        )
        self._reward = torch.empty(
            (self.capacity,), dtype=torch.float32, device=self.device
        )
        self._priorities = torch.ones(
            (self.capacity,), dtype=torch.float32, device=self.device
        )
        self._eps = 1e-6
        self._full = False
        self.idx = 0

    def __add__(self, episode: Episode):
        self.add(episode)
        return self

    def add(self, episode: Episode):
        assert episode.done, "Can only add completed episodes to the replay buffer."
        episode_length = len(episode)
        indices = CircularIndices(
            torch.arange(self.idx, self.idx + episode_length + 1, device=self.device),
            self.capacity,
        )

        self._obs[indices.get()] = episode.obs[: episode_length + 1]
        last_index = indices.pop_last()

        self._action[indices.get()] = episode.action[:episode_length]
        self._opponent_action[indices.get()] = episode.opponent_action[:episode_length]
        self._reward[indices.get()] = episode.reward[:episode_length]

        self._done[indices.get()] = 0.0
        self._done[indices.get()[-1]] = 1.0

        if self._full:
            max_priority = self._priorities.max().to(self.device).item()
        else:
            max_priority = (
                1.0
                if self.idx == 0
                else self._priorities[: self.idx].max().to(self.device).item()
            )

        self._priorities[indices.get()] = max_priority
        self._priorities[last_index] = 0.0

        self.idx = (last_index + 1) % self.capacity
        self._full = self._full or (self.idx < episode_length)

    def update_priorities(self, idxs, priorities):
        self._priorities[idxs] = priorities.squeeze(1).to(self.device) + self._eps

    def sample(self):
        probs = (
            self._priorities if self._full else self._priorities[: self.idx]
        ) ** self.cfg.per_alpha
        probs /= probs.sum()
        total = len(probs)
        idxs = torch.from_numpy(
            np.random.choice(
                total,
                self.cfg.batch_size,
                p=probs.cpu().numpy(),
                replace=not self._full,
            )
        ).to(self.device)
        weights = (total * probs[idxs]) ** (-self.cfg.per_beta)
        weights /= weights.max()

        n_step = self.cfg.get("n_step_returns", 1)
        dim = self.cfg.horizon + n_step
        obs = self._obs[idxs]
        next_obs_shape = self._obs.shape[1:]
        next_obs = torch.empty(
            (dim, self.cfg.batch_size, *next_obs_shape),
            dtype=obs.dtype,
            device=obs.device,
        )
        action = torch.empty(
            (dim, self.cfg.batch_size, *self._action.shape[1:]),
            dtype=torch.float32,
            device=self.device,
        )
        reward = torch.empty(
            (dim, self.cfg.batch_size),
            dtype=torch.float32,
            device=self.device,
        )
        done = torch.empty(
            (dim, self.cfg.batch_size),
            dtype=torch.float32,
            device=self.device,
        )
        truncated_n_step = torch.zeros(
            (dim, self.cfg.batch_size),
            dtype=torch.int32,
            device=self.device,
        )
        reward_n_step_sum = torch.zeros(
            (dim, self.cfg.batch_size),
            dtype=torch.float32,
            device=self.device,
        )

        for t in range(dim - 1, -1, -1):
            _idxs = (idxs + t) % self.capacity
            next_obs[t] = self._obs[(_idxs + 1) % self.capacity]
            action[t] = self._action[_idxs]
            reward[t] = self._reward[_idxs]
            done[t] = self._done[_idxs]

            reward_n_step_sum[t] = reward[t]
            if t != dim - 1:
                truncated_n_step[t] = (truncated_n_step[t + 1] + 1).clamp(
                    max=n_step - 1
                )
                reward_n_step_sum[t] += (
                    self.cfg.discount * reward_n_step_sum[t + 1] * (1 - done[t])
                )
            truncated_n_step[t] *= (1 - done[t]).int()
            if t + n_step < dim:
                reward_n_step_sum -= (
                    (truncated_n_step[t] == n_step - 1).float()
                    * (self.cfg.discount**n_step)
                    * reward[t + n_step]
                )

        return (
            obs,
            next_obs,
            action,
            reward.unsqueeze(2),
            done.unsqueeze(2),
            truncated_n_step,
            reward_n_step_sum.unsqueeze(2),
            idxs,
            weights,
        )


def linear_schedule(schdl, step):
    """
    Outputs values following a linear decay schedule.
    Adapted from https://github.com/facebookresearch/drqv2
    """
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
    raise NotImplementedError(schdl)
