import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import helper as h

class QNetwork(nn.Module):
    """
    Simple feedforward neural network for Q-value approximation.
    """
    def __init__(self, cfg):
        super(QNetwork, self).__init__()
        state_dim = cfg.state_dim
        action_dim = cfg.action_dim
        hidden_dim = cfg.hidden_dim
        self.fc1 = nn.Linear(
                np.array(state_dim).prod() + np.array(action_dim).prod(), hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.fc3 = nn.Linear(cfg.hidden_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

LOG_STD_MAX = 2
LOG_STD_MIN = -5
class Actor(nn.Module):
    """
    Simple feedforward neural network for policy approximation.
    """
    def __init__(self, env, cfg):
        super(Actor, self).__init__()
        state_dim = cfg.state_dim
        action_dim = cfg.action_dim

        self.env = env

        self.cfg = cfg
        self.fc1 = nn.Linear(np.array(state_dim).prod(), cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.activation = nn.ReLU()
        self.fc_mean = nn.Linear(cfg.hidden_dim, np.array(action_dim).prod())
        self.fc_log_std = nn.Linear(cfg.hidden_dim, np.array(action_dim).prod())
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high[:4] - env.action_space.low[:4]) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high[:4] + env.action_space.low[:4]) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))

        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_log_std(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class SACAgent:
    """
    Soft Actor-Critic (SAC) agent implementation.
    """
    def __init__(self, env, cfg):
        self.cfg = cfg
        self.state_dim = cfg.state_dim
        self.action_dim = cfg.action_dim
        self.gamma = cfg.gamma
        self.env = env

        # soft update of target QNetworks
        self.tau = cfg.tau

        # networks
        self.actor = Actor(env, cfg)
        self.qf1 = QNetwork(cfg)
        self.qf2 = QNetwork(cfg)
        self.target_qf1 = deepcopy(self.qf1)
        self.target_qf2 = deepcopy(self.qf2)

        print("networks:")
        print(self.actor)
        print(self.qf1)
        print(self.qf2)

        # alpha for entropy regularization
        # TODO: implement automatic entropy tuning
        self.alpha = cfg.alpha

        # move to device
        self.actor.to(device=self.cfg.device)
        self.qf1.to(device=self.cfg.device)
        self.qf2.to(device=self.cfg.device)
        self.target_qf1.to(device=self.cfg.device)
        self.target_qf2.to(device=self.cfg.device)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.qf1_optimizer = torch.optim.Adam(list(self.qf1.parameters()) + \
                list(self.qf2.parameters()), lr=cfg.critic_lr)

    def update(self, replay_buffer, step=None):
        states, next_states, actions, rewards, dones, idxs, weights = \
                replay_buffer.sample()
        states = h.get_tensor(states, device=self.cfg.device)
        actions = h.get_tensor(actions, device=self.cfg.device)
        rewards = h.get_tensor(rewards, device=self.cfg.device).unsqueeze(1)
        next_states = h.get_tensor(next_states, device=self.cfg.device)
        dones = h.get_tensor(dones, device=self.cfg.device).unsqueeze(1)

        # Critic update
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.get_action(next_states)
            if next_actions.shape[0] == 1:
                next_actions = next_actions.squeeze(0)
            next_qf1_values = self.target_qf1(torch.cat([next_states, next_actions], dim=1))
            next_qf2_values = self.target_qf2(torch.cat([next_states, next_actions], dim=1))
            min_next_qf_values = torch.min(next_qf1_values, next_qf2_values) - self.alpha * next_log_pi
            target_q_values = rewards + (1 - dones) * self.gamma * (min_next_qf_values)

        current_qf1_values = self.qf1(torch.cat([states, actions.float()], dim=1))
        current_qf2_values = self.qf2(torch.cat([states, actions.float()], dim=1))
        critic_loss = F.mse_loss(current_qf1_values, target_q_values) + \
                      F.mse_loss(current_qf2_values, target_q_values)
        self.qf1_optimizer.zero_grad()
        critic_loss.backward()
        self.qf1_optimizer.step()

        # Actor update
        new_actions, log_pi, _ = self.actor.get_action(states)
        qf1_new_actions = self.qf1(torch.cat([states, new_actions], dim=1))
        qf2_new_actions = self.qf2(torch.cat([states, new_actions], dim=1))
        min_qf_new_actions = torch.min(qf1_new_actions, qf2_new_actions)
        actor_loss = (self.alpha * log_pi - min_qf_new_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        loss = critic_loss + actor_loss
        # gradient norm
        grad_norm = 0.0
        for p in list(self.qf1.parameters()) + list(self.qf2.parameters()) + list(self.actor.parameters()):
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2

        # Soft update target networks if it's time
        if step % self.cfg.target_update_freq == 0:
            self.soft_update_target_network()

        # PER priority update
        with torch.no_grad():
            current_q_values = torch.min(current_qf1_values, current_qf2_values)
            priority_loss = torch.abs(current_q_values - target_q_values)
            replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4))

        return {'total_loss': loss.item(),
                'actor_loss': actor_loss.item(),
                'critic_loss': critic_loss.item(),
                'grad_norm': float(grad_norm)}

    @torch.no_grad()
    def act(self, x):
        x = h.get_tensor(x, device=self.cfg.device)
        actions, _, _ = self.actor.get_action(x)
        return actions.cpu().numpy()

    @torch.no_grad()
    def plan(self, x, eval_mode=False, step=None, t0=True):
        x = h.get_tensor(x, device=self.cfg.device)
        batch_dim_missing = False
        if np.prod(x.shape) == self.state_dim:
            x = x.unsqueeze(0)
            batch_dim_missing = True
        out = self.act(x)
        if batch_dim_missing:
            out = out[0]
        return out

    @torch.no_grad()
    def soft_update_target_network(self):
        for target_param, param in zip(self.target_qf1.parameters(), self.qf1.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
        for target_param, param in zip(self.target_qf2.parameters(), self.qf2.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def save(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'qf1_state_dict': self.qf1.state_dict(),
            'qf2_state_dict': self.qf2.state_dict(),
            'target_qf1_state_dict': self.target_qf1.state_dict(),
            'target_qf2_state_dict': self.target_qf2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'qf1_optimizer_state_dict': self.qf1_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.cfg.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.qf1.load_state_dict(checkpoint['qf1_state_dict'])
        self.qf2.load_state_dict(checkpoint['qf2_state_dict'])
        self.target_qf1.load_state_dict(checkpoint['target_qf1_state_dict'])
        self.target_qf2.load_state_dict(checkpoint['target_qf2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.qf1_optimizer.load_state_dict(checkpoint['qf1_optimizer_state_dict'])
