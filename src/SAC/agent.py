import os
from pathlib import Path
import torch as T
import torch.nn.functional as F
import numpy as np
import helper
from memory import ReplayBuffer, PrioritizedReplayBuffer
from network import ActorNet, CriticNet
import pickle

class Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gamma = cfg.gamma
        self.buffer_type = cfg.get('buffer_type', 'replay')
        if self.buffer_type == 'per':
            self.memory = PrioritizedReplayBuffer(cfg.buffer_max_size)
        else:
            self.memory = ReplayBuffer(cfg.buffer_max_size, cfg.input_dims, cfg.n_actions)
        self.batch_size = cfg.batch_size
        self.n_actions = cfg.n_actions
        self.scale = cfg.reward_scale

        # initialize actor and critic networks
        self.actor = ActorNet(cfg.lr_actor, cfg.input_dims, n_actions=cfg.n_actions,
                              hidden_size=cfg.hidden_size, max_action=[1] * cfg.n_actions)
        self.critic_1 = CriticNet(cfg.lr_critic, cfg.input_dims, n_actions=cfg.n_actions,
                              hidden_size=cfg.hidden_size)
        self.critic_2 = CriticNet(cfg.lr_critic, cfg.input_dims, n_actions=cfg.n_actions,
                              hidden_size=cfg.hidden_size)

        # target networks
        self.critic_1_target = CriticNet(cfg.lr_critic, cfg.input_dims, n_actions=cfg.n_actions,
                              hidden_size=cfg.hidden_size)
        self.critic_2_target = CriticNet(cfg.lr_critic, cfg.input_dims, n_actions=cfg.n_actions,
                                hidden_size=cfg.hidden_size)
        self.update_target_networks(method='hard')
        self.tau = cfg.tau
        self.target_update_freq = cfg.target_update_freq

        if getattr(cfg, 'resume', False):
            if not self.use_most_recent_models():
                raise ValueError

        # entropy coefficient
        if cfg.automatic_entropy_tuning:
            if not hasattr(self, 'log_alpha'):
                if cfg.alpha:
                    self.log_alpha = T.tensor(np.log(cfg.alpha), requires_grad=True).to(self.actor.device)
                else:
                    self.log_alpha = T.zeros(1, requires_grad=True, device=self.actor.device)
            self.alpha_optim = T.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)
            self.target_entropy = -np.prod(cfg.n_actions).item()
        else:
            self.alpha = T.tensor(cfg.alpha).to(self.actor.device)


    def use_most_recent_models(self):
        """Loads the most recent models from the results directory."""
        results_dir = Path(__file__).resolve().parent / "results" / self.cfg.exp_name / 'models'
        best_model_dir = helper.get_latest_checkpoint(results_dir)
        if best_model_dir is not None:
            print("Resuming from checkpoint:", best_model_dir)
            self.load_models(best_model_dir)
            return True
        else:
            print(f"No checkpoint found at {best_model_dir} to resume from.")
            return False

    def get_alpha(self):
        """Returns the current value of the entropy coefficient alpha."""
        if hasattr(self, 'log_alpha'):
            return self.log_alpha.exp()
        else:
            return self.alpha


    def update_target_networks(self, method='soft'):
        """Updates the target networks with the current critic networks' parameters."""
        if method == 'hard':
            self.critic_1_target.load_state_dict(self.critic_1.state_dict())
            self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        elif method == 'soft':
            for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            raise ValueError("Invalid update type. Use 'soft' or 'hard'.")


    def get_models(self):
        return self.actor, self.critic_1, self.critic_2

    def choose_action(self, observation, step=None):
        """Chooses an action based on the current policy (actor network)."""
        if step is not None and step < self.cfg.warmup_games:
            return np.random.uniform(low=-1, high=1, size=self.n_actions)
        state = T.from_numpy(observation).float().unsqueeze(0).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]

    def plan(self, observation, eval_mode=False, step=None):
        """Alias for choose_action"""
        return self.choose_action(observation)

    def act(self, x):
        """Alias for choose_action"""
        return self.choose_action(x)

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
        if hasattr(self, 'log_alpha'):
            file_path_alpha = os.path.join(folder_path, 'log_alpha.pth')
            T.save(self.log_alpha, file_path_alpha)

    def load_models(self, folder_path):
        """Loads the parameters of the actor and critic networks."""
        print('loading models ..')
        file_path_actor = os.path.join(folder_path, 'actor.pth')
        self.actor.load(file_path_actor)
        file_path_critic1 = os.path.join(folder_path, 'critic_1.pth')
        self.critic_1.load(file_path_critic1)
        file_path_critic2 = os.path.join(folder_path, 'critic_2.pth')
        self.critic_2.load(file_path_critic2)
        self.update_target_networks(method='hard')
        if hasattr(self, 'log_alpha') or self.cfg.automatic_entropy_tuning:
            file_path_alpha = os.path.join(folder_path, 'log_alpha.pth')
            # check if file exists
            if os.path.exists(file_path_alpha):
                self.log_alpha = T.load(file_path_alpha, map_location=self.actor.device, weights_only=True)
            elif self.cfg.alpha:
                self.log_alpha = T.tensor(np.log(self.cfg.alpha), requires_grad=True).to(self.actor.device)
            else:
                raise ValueError(f"No alpha file / value found to load from at {file_path_alpha} / config yaml.")
            self.log_alpha = self.log_alpha.clone().detach().requires_grad_(True)

    def learn(self, step=None):
        """Updates the networks (actor, critics, and alpha) based on sampled experiences."""
        log_data = {}
        if self.memory.mem_cntr < self.batch_size:
            return log_data

        if step is not None and step < self.cfg.warmup_games:
            return log_data

        if self.buffer_type == 'per':
            state, action, reward, new_state, done, indices, weights = self.memory.sample_buffer(self.batch_size)
            weights = T.tensor(weights, dtype=T.float).to(self.actor.device)
        else:
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.as_tensor(np.asarray(action), dtype=T.float, device=self.actor.device)


        # Compute target Q-values
        with T.no_grad():
            next_actions, next_log_probs = self.actor.sample_normal(state_, reparameterize=False)
            q1_next = self.critic_1_target.forward(state_, next_actions)
            q2_next = self.critic_2_target.forward(state_, next_actions)
            q_next = T.min(q1_next, q2_next) - self.get_alpha() * next_log_probs
            q_target = self.scale * reward + self.gamma * (1 - done.float()) * q_next.view(-1)

        # Update critics
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q1_old = self.critic_1.forward(state, action).view(-1)
        q2_old = self.critic_2.forward(state, action).view(-1)
        if self.buffer_type == 'per':
            critic_1_loss = (0.5 * F.mse_loss(q1_old, q_target, reduction='none') * weights).mean()
            critic_2_loss = (0.5 * F.mse_loss(q2_old, q_target, reduction='none') * weights).mean()
            priorities = (0.5 * (T.abs(q1_old - q_target) + T.abs(q2_old - q_target))).cpu().detach().numpy() + 1e-6
        else:
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
        actor_loss = (self.get_alpha().detach() * log_probs.view(-1) - critic_value).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Update alpha
        if hasattr(self, 'log_alpha'):
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            log_data.update({'Losses/alpha_loss': alpha_loss.item()})

        # Update target networks
        if step is not None and step % self.target_update_freq == 0:
            self.update_target_networks(method='soft')

        # Update Replay Buffer priorities if using PER
        if self.buffer_type == 'per':
            self.memory.update_priorities(indices, priorities)
            log_data.update({
                'per/avg_priority': np.mean(priorities),
                'per/max_priority': np.max(priorities),
                'hist:per/priority_distribution': priorities,
                'per/mean_weight': np.mean(weights.cpu().numpy()),
                'per/max_weight': np.max(weights.cpu().numpy())
            })

        log_data.update({
            'Losses/actor_loss': actor_loss.item(),
            'Losses/critic_1_loss': critic_1_loss.item(),
            'Losses/critic_2_loss': critic_2_loss.item(),
            'Metrics/entropy': -log_probs.mean().item(),
            'HyperParam/alpha': self.get_alpha().item()
        })
        return log_data
