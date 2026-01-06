import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import helper as h
from action_space import CustomActionSpace

class Network(nn.Module):
	def __init__(self, input_dim, output_dim, hidden_layers=1, hidden_dim=128):
		super(Network, self).__init__()
		self.fc1 = nn.Linear(input_dim, hidden_dim)
		layers = []
		for _ in range(hidden_layers):
			layers.append(nn.Linear(hidden_dim, hidden_dim))
			layers.append(nn.ReLU())
		self.fc2 = nn.Sequential(*layers)
		self.fc3 = nn.Linear(hidden_dim, output_dim)
		print(self)

	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = self.fc2(x)
		return self.fc3(x)

	def save(self, filepath):
		torch.save(self.state_dict(), filepath)

	def get_device(self):
		return next(self.parameters()).device

	def load(self, filepath):
		self.load_state_dict(torch.load(filepath, weights_only=True, map_location=self.get_device()))
		self.eval()

class DQNAgent:
	def __init__(self, cfg):
		self.cfg = cfg
		self.state_dim = cfg.state_dim
		self.action_dim = cfg.action_dim
		self.gamma = cfg.gamma
		self.tau = cfg.tau

		self.policy_net = Network(self.state_dim, self.action_dim,
							hidden_layers=cfg.hidden_layer_n,
							hidden_dim=cfg.hidden_dim)
		self.target_net = deepcopy(self.policy_net)
		self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
		self.loss_fn = lambda a, b, w: ((a - b) ** 2 * w).mean()

		# epsilon for epsilon-greedy action selection
		self.eps = h.linear_schedule(self.cfg.eps_schedule, 0)

		# move to device
		self.policy_net.to(device=cfg.device)
		self.target_net.to(device=cfg.device)

	def update(self, replay_buffer, step=None):
		states, next_states, actions, rewards, dones, idxs, weights = \
				replay_buffer.sample()
		states = torch.tensor(states, dtype=torch.float32).to(device=self.cfg.device)
		actions = torch.tensor(actions, dtype=torch.long).to(device=self.cfg.device)
		rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device=self.cfg.device)
		next_states = torch.tensor(next_states, dtype=torch.float32).to(device=self.cfg.device)
		dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device=self.cfg.device)

		actions = actions.argmax(dim=1, keepdim=True)
		current_q_values = self.policy_net(states).gather(1, actions)
		with torch.no_grad():
			next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
			target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

		loss = self.loss_fn(current_q_values, target_q_values.detach(),
					  torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device=self.cfg.device))
		self.optimizer.zero_grad()
		loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),
											 self.cfg.grad_clip_norm)
		self.optimizer.step()
		priority_loss = h.l1(current_q_values, target_q_values, 1 - dones)
		replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())

		self.soft_update_target_network()
		if step is not None:
			self.eps = h.linear_schedule(self.cfg.eps_schedule, step)

		return {'total_loss': loss.item(),
		  	    'grad_norm': float(grad_norm),
				'eps': self.eps}

	@torch.no_grad()
	def act(self, x):
		x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
		x = x.to(device=self.cfg.device)
		with torch.no_grad():
			q_values = self.policy_net.forward(x)
		q_max_idx = q_values.argmax().item()
		action = CustomActionSpace().discrete_to_continuous(q_max_idx)
		return action, q_max_idx

	@torch.no_grad()
	def plan(self, x, eval_mode=False, step=None, t0=True):
		if eval_mode or np.random.rand() > self.eps:
			return self.act(x)
		else:
			q_max_idx = np.random.randint(self.action_dim)
			action = CustomActionSpace().discrete_to_continuous(q_max_idx)
			return action, q_max_idx

	@torch.no_grad()
	def soft_update_target_network(self):
		for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
			target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

	def save(self, filepath):
		self.policy_net.save(filepath)

	def load(self, filepath):
		self.policy_net.load(filepath)
		self.target_net = deepcopy(self.policy_net)
