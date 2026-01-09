import os
import torch as T
import torch.nn.functional as F
import numpy as np
from memory import ReplayBuffer
from network import ActorNet, CriticNet
import pickle

class Agent:
    def __init__(self, lr_actor=0.0003, lr_critic=0.0003, input_dims=[18], env=None, gamma=0.99, n_actions=4, buffer_max_size=1000000,
                 hidden_size=256, batch_size=256, reward_scale=2, alpha=0.1):
        self.env = env
        self.gamma = gamma
        self.memory = ReplayBuffer(buffer_max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.scale = reward_scale

        # initialize actor and critic networks
        self.actor = ActorNet(lr_actor, input_dims, n_actions=n_actions,
                              hidden_size=hidden_size, max_action=[1] * n_actions)
        self.critic_1 = CriticNet(lr_critic, input_dims, n_actions=n_actions,
                              hidden_size=hidden_size)
        self.critic_2 = CriticNet(lr_critic, input_dims, n_actions=n_actions,
                              hidden_size=hidden_size)

        self.alpha = alpha  # entropy coefficient

    def get_models(self):
        return self.actor, self.critic_1, self.critic_2

    def choose_action(self, observation):
        """Chooses an action based on the current policy (actor network)."""
        state = T.from_numpy(observation).float().unsqueeze(0).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]

    def plan(self, observation, eval_mode=False, step=None):
        """Alias for choose_action"""
        return self.choose_action(observation)

    def store(self, state, action, reward, new_state, done):
        """Stores a transition in the replay buffer."""
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self, folder_path):
        """Saves the parameters of the actor and critic networks."""
        print('Saving models and optimizer states...')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path_actor = os.path.join(folder_path, 'actor.pth')
        file_path_critic1 = os.path.join(folder_path, 'critic_1.pth')
        file_path_critic2 = os.path.join(folder_path, 'critic_2.pth')
        self.actor.save(file_path_actor)
        self.critic_1.save(file_path_critic1)
        self.critic_2.save(file_path_critic2)

    def load_models(self, file_path_actor=None, file_path_critic1=None, file_path_critic2=None):
        """Loads the parameters of the actor and critic networks."""
        print('loading models ..')
        self.actor.load(file_path_actor)
        self.critic_1.load(file_path_critic1)
        self.critic_2.load(file_path_critic2)

    def learn(self, step=None):
        """Updates the networks (actor, critics, and alpha) based on sampled experiences."""
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        # Compute target Q-values
        with T.no_grad():
            next_actions, next_log_probs = self.actor.sample_normal(state_, reparameterize=False)
            q1_next = self.critic_1.forward(state_, next_actions)
            q2_next = self.critic_2.forward(state_, next_actions)
            q_next = T.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = self.scale * reward + self.gamma * (1 - done.float()) * q_next.view(-1)

        # Update critics
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q1_old = self.critic_1.forward(state, action).view(-1)
        q2_old = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old, q_target)
        critic_2_loss = 0.5 * F.mse_loss(q2_old, q_target)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # Update actor
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        q1_new = self.critic_1.forward(state, actions)
        q2_new = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new, q2_new).view(-1)
        actor_loss = (self.alpha * log_probs.view(-1) - critic_value).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()


        return {
            'actor_loss': actor_loss.item(),
            'critic_1_loss': critic_1_loss.item(),
            'critic_2_loss': critic_2_loss.item(),
            'alpha': self.alpha
        }
