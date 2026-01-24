import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np


class ActorNet(nn.Module):
    def __init__(self, lr_actor, input_dims, max_action, hidden_size=256, n_actions=2):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_size = hidden_size
        self.n_actions = n_actions
        self.max_action = max_action
        self.reparam_noise = 1e-6  # noise added during reparameterization

        self.fc1 = nn.Linear(*self.input_dims, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.mu = nn.Linear(self.hidden_size, self.n_actions)
        self.sigma = nn.Linear(self.hidden_size, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_actor)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)  # for numerical stability
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        tanh_actions = T.tanh(actions)
        action = tanh_actions * T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)

        # log_prob corrections for squashing and scaling (hopefully corrects, non-unity scaling)
        # was working before, as max_action is 1, but should be better this way, lol, thanks M4ML
        # y = a * tanh(x)
        # dy/dx = a * (1 - tanh^2(x))
        # log_p(y) = log_p(x) - log|det(dy/dx)|
        # log_p(y) = log_p(x) - (log(a) + log(1 - tanh^2(x)))
        log_probs -= T.log(1 - tanh_actions.pow(2) + self.reparam_noise)
        log_probs -= T.log(T.tensor(self.max_action).to(self.device))
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save(self, file_path):
        T.save(self.state_dict(), file_path)

    def load(self, file_path):
        self.load_state_dict(T.load(file_path, weights_only=False, map_location=self.device))


class CriticNet(nn.Module):
    def __init__(self, lr_critic, input_dims, n_actions, hidden_size=256):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_size = hidden_size
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_critic)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = self.fc1(T.cat([state, action], dim=1))
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.q(x)

    def save(self, file_path):
        T.save(self.state_dict(), file_path)

    def load(self, file_path):
        self.load_state_dict(T.load(file_path, weights_only=False, map_location=self.device))
