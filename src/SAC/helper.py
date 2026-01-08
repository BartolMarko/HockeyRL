import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from torch.utils.tensorboard import SummaryWriter
import wandb

__REDUCE__ = lambda b: 'mean' if b else 'none'

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

def get_tensor(x, device, dtype=torch.float32):
	"""Converts input to a torch tensor on the specified device."""
	if isinstance(x, list):
		x = np.array(x)
	if isinstance(x, np.ndarray):
		x = torch.tensor(x)
	if isinstance(x, torch.Tensor):
		x = x.to(device=device, dtype=dtype)
	return x

def orthogonal_init(m):
	"""Orthogonal layer initialization."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if m.bias is not None:
			nn.init.zeros_(m.bias)
	elif isinstance(m, nn.Conv2d):
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data, gain)
		if m.bias is not None:
			nn.init.zeros_(m.bias)


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
		eps = _standard_normal(shape,
							   dtype=self.loc.dtype,
							   device=self.loc.device)
		eps *= self.scale
		if clip is not None:
			eps = torch.clamp(eps, -clip, clip)
		x = self.loc + eps
		return self._clamp(x)


class NormalizeImg(nn.Module):
	"""Normalizes pixel observations to [0,1) range."""
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.)


class Flatten(nn.Module):
	"""Flattens its input to a (batched) vector."""
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)


def enc(cfg):
	"""Returns a TOLD encoder."""
	layers = [nn.Linear(cfg.obs_shape[0], cfg.enc_dim), nn.ELU(),
		   		nn.Linear(cfg.enc_dim, cfg.latent_dim)]
	return nn.Sequential(*layers)


def mlp(in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
	"""Returns an MLP."""
	if isinstance(mlp_dim, int):
		mlp_dim = [mlp_dim, mlp_dim]
	return nn.Sequential(
		nn.Linear(in_dim, mlp_dim[0]), act_fn,
		nn.Linear(mlp_dim[0], mlp_dim[1]), act_fn,
		nn.Linear(mlp_dim[1], out_dim)
	)

def q(cfg, act_fn=nn.ELU()):
	"""Returns a Q-function that uses Layer Normalization."""
	return nn.Sequential(nn.Linear(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim), nn.LayerNorm(cfg.mlp_dim), nn.Tanh(),
						 nn.Linear(cfg.mlp_dim, cfg.mlp_dim), nn.ELU(),
						 nn.Linear(cfg.mlp_dim, 1))


class Episode(object):
	"""Storage object for a single episode."""
	def __init__(self, cfg, init_obs):
		self.cfg = cfg
		self.device = torch.device(cfg.device)
		dtype = torch.float32
		max_episode_length = cfg.max_episode_length
		self.obs = torch.empty((max_episode_length + 1, *init_obs.shape), dtype=dtype, device=self.device)
		self.obs[0] = torch.tensor(init_obs, dtype=dtype, device=self.device)
		self.action = torch.empty((max_episode_length, cfg.action_dim), dtype=torch.float32, device=self.device)
		self.opponent_action = torch.empty((max_episode_length, cfg.opponent_action_dim), dtype=dtype, device=self.device)
		self.reward = torch.zeros((max_episode_length,), dtype=torch.float32, device=self.device)
		self.done = False
		self.length = 0

	def __len__(self):
		return self.length

	@property
	def first(self):
		return len(self) == 0

	def __add__(self, transition):
		self.add(*transition)
		return self

	def add(self, obs, action, opponent_action, reward, done):
		assert not self.done, "Episode has terminated. Can not add more transitions."
		assert self.length < self.cfg.max_episode_length, "Episode buffer is full."
		self.obs[self.length + 1] = torch.tensor(obs, dtype=self.obs.dtype, device=self.obs.device)
		self.action[self.length] = torch.tensor(action, dtype=self.action.dtype, device=self.action.device)
		self.opponent_action[self.length] = torch.tensor(opponent_action, dtype=self.opponent_action.dtype, device=self.opponent_action.device)
		self.reward[self.length] = reward
		self.done = done
		self.length += 1


class CircularIndices():
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


class ReplayBuffer():
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
		self._obs = torch.empty((self.capacity+1, *obs_shape), dtype=dtype, device=self.device)
		self._action = torch.empty((self.capacity, cfg.action_dim), dtype=torch.float32, device=self.device)
		self._opponent_action = torch.empty((self.capacity, cfg.opponent_action_dim), dtype=dtype, device=self.device)
		self._done = torch.empty((self.capacity,), dtype=torch.float32, device=self.device)
		self._reward = torch.empty((self.capacity,), dtype=torch.float32, device=self.device)
		self._priorities = torch.ones((self.capacity,), dtype=torch.float32, device=self.device)
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

		self._obs[indices.get()] = episode.obs[:episode_length + 1]
		last_index = indices.pop_last()

		self._action[indices.get()] = episode.action[:episode_length]
		self._opponent_action[indices.get()] = episode.opponent_action[:episode_length]
		self._reward[indices.get()] = episode.reward[:episode_length]

		self._done[indices.get()] = 0.0
		self._done[indices.get()[-1]] = 1.0

		if self._full:
			max_priority = self._priorities.max().to(self.device).item()
		else:
			max_priority = 1. if self.idx == 0 else self._priorities[:self.idx].max().to(self.device).item()

		self._priorities[indices.get()] = max_priority
		self._priorities[last_index] = 0.0

		self.idx = (last_index + 1) % self.capacity
		self._full = self._full or (self.idx < episode_length)

	def update_priorities(self, idxs, priorities):
		self._priorities[idxs] = priorities.squeeze(1).to(self.device) + self._eps

	def sample(self):
		probs = (self._priorities if self._full else self._priorities[:self.idx]) ** self.cfg.per_alpha
		probs /= probs.sum()
		total = len(probs)
		idxs = torch.from_numpy(np.random.choice(total, self.cfg.batch_size, p=probs.cpu().numpy(), replace=not self._full)).to(self.device)
		weights = (total * probs[idxs]) ** (-self.cfg.per_beta)
		weights /= weights.max()

		obs = self._obs[idxs]
		next_obs = self._obs[idxs + 1]
		action = self._action[idxs]
		reward = self._reward[idxs]
		done = self._done[idxs]

		if not action.is_cuda and self.device != torch.device('cpu'):
			action, reward, done, idxs, weights = \
				action.cuda(), reward.cuda(), done.cuda(), idxs.cuda(), weights.cuda()

		return obs, next_obs, action, reward, done, idxs, weights


def linear_schedule(schdl, step):
	"""
	Outputs values following a linear decay schedule.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	try:
		return float(schdl)
	except ValueError:
		match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
		if match:
			init, final, duration = [float(g) for g in match.groups()]
			mix = np.clip(step / duration, 0.0, 1.0)
			return (1.0 - mix) * init + mix * final
	raise NotImplementedError(schdl)

class Logger:
	"""
	Simple logger class to store training statistics.
	Logs a dictionary for tensorboard logging and if cfg provides wandb, use that as well.
	"""
	def __init__(self, cfg, project_dir=None):
		self.cfg = cfg
		self.tb_logger = SummaryWriter(project_dir / 'logs') if project_dir is not None else None
		if self.cfg.get('use_wandb', False):
			import os
			if os.getenv('WANDB_API_KEY') is None:
				raise ValueError("WANDB_API_KEY environment variable not set.")
			print("wandb:")
			print(" - Project:", self.cfg.get('wandb_project'))
			self.wandb = wandb.init(project=self.cfg.get('wandb_project'),
						   config=dict(cfg), monitor_gym=True, sync_tensorboard=True)
		else:
			self.wandb = None

		assert self.tb_logger is not None or self.wandb is not None, "No logging method specified."

		self.data = {}

	def add_scalar(self, key, value, step):
		if self.tb_logger is not None:
			self.tb_logger.add_scalar(key, value, step)
		if self.wandb is not None:
			self.wandb.log({key: value}, step=step)
		self.data[key] = (step, value)

	def add_gif(self, key, gif_path, step):
		if self.wandb is not None:
			self.wandb.log({key: wandb.Video(gif_path, caption=f"step: {step}")}, step=step)

	def get_logs(self):
		return self.data

	def clear(self):
		self.data = {}

	def log_git_info(self, filename='git-commit-hash.txt'):
		commit_info = read_commit_info(filename)
		# Log git commit info as text information
		if self.tb_logger is not None:
			self.tb_logger.add_text('git-commit-hash', commit_info['git-commit-hash'], 0)
			self.tb_logger.add_text('git-commit-log', commit_info['git-commit-log'], 0)
		if self.wandb is not None:
			self.wandb.config.update({
				'git-commit-hash': commit_info['git-commit-hash'],
				'git-commit-log': commit_info['git-commit-log']
			})

def read_commit_info(filename='git-commit-hash.txt'):
	"""Reads git commit info from a file."""
	commit_info = {}
	try:
		with open(filename, 'r') as f:
			for line in f:
				key, value = line.strip().split(': ', 1)
				commit_info[key] = value
	except FileNotFoundError:
		commit_info = {
			'git-commit-hash': 'N/A',
			'git-commit-log': 'N/A'
		}
	return commit_info
