import os
from pathlib import Path
import torch as T
import torch.nn.functional as F
import numpy as np
from memory import get_memory_buffer
from exploration import RandomExplorer, GaussianExplorer, OrnsteinUhlenbeckExplorer, CuriousExplorer, OptimisticExplorer
from network import ActorNet, CriticNet
import pickle

class Agent:
    def __init__(self, cfg, inference_only=False):
        self.cfg = cfg
        self.name = cfg.exp_name
        self.gamma = cfg.gamma
        self.inference_only = inference_only
        if inference_only:
            self.buffer_type = 'none'
            self.memory = None
        else:
            self.buffer_type = cfg.get('buffer_type', 'replay')
            self.memory = get_memory_buffer(cfg)

        self.batch_size = cfg.batch_size
        self.n_actions = cfg.n_actions

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

        # exploration strategy
        explorer_cfg = cfg.get('explorer', {'type': 'random'})
        explorer_type = explorer_cfg.get('type', 'random')
        if explorer_type == 'random':
            self.explorer = RandomExplorer(cfg.n_actions)
        elif explorer_type == 'gaussian':
            mu = explorer_cfg.get('mu', 0)
            std_dev = explorer_cfg.get('std_dev', 0.8)
            self.explorer = GaussianExplorer(cfg.n_actions, mu=mu, sigma=std_dev)
        elif explorer_type == 'ou':
            theta = explorer_cfg.get('theta', 0.15)
            sigma = explorer_cfg.get('sigma', 0.2)
            dt = explorer_cfg.get('dt', 1e-2)
            self.explorer = OrnsteinUhlenbeckExplorer(cfg.n_actions, theta=theta, sigma=sigma, dt=dt)
        elif explorer_type == 'optimistic':
            self.explorer = OptimisticExplorer(self, explorer_cfg)
        elif explorer_type == 'curious':
            self.explorer = CuriousExplorer(self, explorer_cfg)
        else:
            raise ValueError(f"Unknown explorer type: {explorer_type}")

        if getattr(cfg, 'resume', False):
            if not self.use_most_recent_models():
                raise ValueError

        # entropy coefficient
        if cfg.automatic_entropy_tuning:
            if not hasattr(self, 'log_alpha'):
                if hasattr(cfg, 'alpha') and cfg.alpha:
                    self.log_alpha = T.tensor(np.log(cfg.alpha), requires_grad=True).to(self.actor.device)
                else:
                    self.log_alpha = T.zeros(1, requires_grad=True, device=self.actor.device)
            self.alpha_optim = T.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)
            self.target_entropy = -np.prod(cfg.n_actions).item()
        else:
            if hasattr(cfg, 'alpha') and cfg.alpha:
                self.alpha = T.tensor(cfg.alpha).to(self.actor.device)
            else:
                print("[WARN] No alpha value specified in config for fixed alpha. Using default alpha=0.2")
                self.alpha = T.tensor(0.2).to(self.actor.device)

        # vectorized env support flag
        num_envs = cfg.get('num_envs', 1)
        self.is_vectorized = num_envs > 1
        self.explorer.support_vec_env(num_envs)

    def train(self):
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        self.explorer.train()

    def eval(self):
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        self.explorer.eval()

    def show_info(self):
        print("Agent Configuration:")
        print(f"  Algorithm: Soft Actor-Critic (SAC)")
        print(f"  Replay Buffer Type: {self.buffer_type}")
        if self.buffer_type in ['per', 'n-step-per']:
            print(f"    Prioritized Experience Replay: Enabled")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Discount Factor (Gamma): {self.gamma}")
        print(f"  Target Network Update Frequency: {self.target_update_freq}")
        print(f"  Tau (Soft Update Coefficient): {self.tau}")
        if hasattr(self, 'log_alpha'):
            print(f"  Automatic Entropy Tuning: Enabled")
            print(f"    Initial Alpha (Entropy Coefficient): {self.get_alpha().item()}")
            print(f"    Target Entropy: {self.target_entropy}")
        else:
            print(f"  Automatic Entropy Tuning: Disabled")
            print(f"    Fixed Alpha (Entropy Coefficient): {self.alpha.item()}")
        print(f"  Explorer Type: {self.explorer.id()}")
        print(f"  Vectorized Environment Support: {'Enabled' if self.is_vectorized else 'Disabled'}")
        if self.is_vectorized:
            print(f"    Number of Environments: {self.cfg.get('num_envs', 1)}")


    def end_episode(self):
        self.explorer.end_episode()
        if self.memory is not None:
            self.memory.flush()

    def use_most_recent_models(self):
        """Loads the most recent models from the results directory."""
        from helper import get_latest_checkpoint
        results_dir = Path(__file__).resolve().parent / "results" / self.cfg.exp_name / 'models'
        best_model_dir = get_latest_checkpoint(results_dir)
        if best_model_dir is not None:
            # remove till the results part
            model_dir_striped = str(best_model_dir).split('results' + os.sep)[-1]
            print("[INFO] Resuming from checkpoint:", model_dir_striped)
            self.load_models(best_model_dir)
            return True
        else:
            print(f"[WARN] No checkpoint found from {results_dir} to resume from.")
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

    def reset_explorer(self):
        """Resets the exploration strategy."""
        self.explorer.reset()

    def choose_action(self, observation, step=None, eval_mode=False):
        """Chooses an action based on the current policy (actor network)."""
        if eval_mode:
             return self.choose_action_from_policy(observation)
        warmup = (step is not None) and (step < self.cfg.warmup_games)
        return self.explorer.choose_action(observation, agent=self, step=step, warmup=warmup)

    def choose_action_batch(self, observation, step=None, eval_mode=False):
        """Batched observation version of choose_action for vectorized envs"""
        if eval_mode:
             return self.choose_action_from_policy_batch(observation)
        warmup = (step is not None) and (step < self.cfg.warmup_games)
        return self.explorer.choose_action_batch(observation, agent=self, step=step, warmup=warmup)

    def plan(self, observation, eval_mode=True, step=None):
        """Alias for choose_action with eval_mode defaulting to True"""
        return self.choose_action(observation, step=step, eval_mode=eval_mode)

    def plan_batch(self, observation, eval_mode=True, step=None):
        """Batched observation version of plan / choose_action for vectorized envs"""
        return self.choose_action_batch(observation, step=step, eval_mode=eval_mode)

    def act(self, x):
        """Alias for choose_action with eval_mode set to True"""
        return self.choose_action(x, eval_mode=True)

    def choose_action_from_policy(self, observation):
        """Helper for explorers to get the raw policy action"""
        state = T.from_numpy(observation).float().unsqueeze(0).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]

    def choose_action_from_policy_batch(self, observation):
        """Batched observation version of choose_action_from_policy for vectorized envs"""
        try:
            state = T.from_numpy(observation).float().to(self.actor.device)
        except TypeError:
            observation = np.array([obs for obs in observation])
            state = T.from_numpy(observation).float().to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()

    def store(self, state, action, reward, new_state, done):
        """Stores a transition in the replay buffer, adding intrinsic reward (curious)."""
        intrinsic_reward = self.explorer.get_intrinsic_reward(state, action, new_state)
        total_reward = reward + intrinsic_reward
        self.memory.store_transition(state, action, total_reward, new_state, done)

    def save_models(self, folder_path, memory=False):
        """Saves the parameters of the actor and critic networks."""
        print('[SAVE] Saving models and optimizer states...')
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
        if memory:
            memory_filename = os.path.join(folder_path, 'buffer')
            self.memory.save(memory_filename)

    def load_models(self, folder_path, memory=False):
        """Loads the parameters of the actor and critic networks."""
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
            elif hasattr(self.cfg, 'alpha') and self.cfg.alpha:
                self.log_alpha = T.tensor(np.log(self.cfg.alpha), requires_grad=True).to(self.actor.device)
            else:
                print(f"[WARN] No alpha file / value found to load from at {file_path_alpha} / config yaml. Using default alpha=0.2")
                self.log_alpha = T.tensor(np.log(0.2), requires_grad=True).to(self.actor.device)
            self.log_alpha = self.log_alpha.clone().detach().requires_grad_(True)
        if memory:
            memory_filename = os.path.join(folder_path, 'buffer')
            if os.path.exists(memory_filename):
                self.memory.load(memory_filename)
            else:
                print(f"[WARN] No memory buffer file found to load from at {memory_filename}.")

    def learn(self, step=None):
        """Updates the networks (actor, critics, and alpha) based on sampled experiences."""
        log_data = {}
        if self.memory.mem_cntr < self.batch_size:
            return log_data

        if self.buffer_type in [ 'per', 'n-step-per' ]:
            state, action, reward, new_state, done, indices, weights = self.memory.sample_buffer(self.batch_size)
            weights = T.tensor(weights, dtype=T.float).to(self.actor.device)
        else:
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float, device=self.actor.device)

        # Compute target Q-values
        with T.no_grad():
            next_actions, next_log_probs = self.actor.sample_normal(state_, reparameterize=False)
            q1_next = self.critic_1_target.forward(state_, next_actions)
            q2_next = self.critic_2_target.forward(state_, next_actions)
            q_next = T.min(q1_next, q2_next) - self.get_alpha() * next_log_probs
            if self.buffer_type == 'n-step-per':
                # use gamma^n for skipping n-1 steps
                q_target = reward + ( self.gamma ** self.cfg.n_step_buffer_n ) * (1 - done.float()) * q_next.view(-1)
            else:
                q_target = reward + self.gamma * (1 - done.float()) * q_next.view(-1)

        # Update critics
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q1_old = self.critic_1.forward(state, action).view(-1)
        q2_old = self.critic_2.forward(state, action).view(-1)
        if self.buffer_type in ['per', 'n-step-per']:
            critic_1_loss = (0.5 * F.mse_loss(q1_old, q_target, reduction='none') * weights).mean()
            critic_2_loss = (0.5 * F.mse_loss(q2_old, q_target, reduction='none') * weights).mean()
            priorities = (0.5 * (T.abs(q1_old - q_target) + T.abs(q2_old - q_target))).cpu().detach().numpy()
            priorities = np.clip(priorities, 1e-6, 1000.0)
        else:
            critic_1_loss = 0.5 * F.mse_loss(q1_old, q_target)
            critic_2_loss = 0.5 * F.mse_loss(q2_old, q_target)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()

        critic1_grad_norm = sum(p.grad.norm().item() for p in self.critic_1.parameters() if p.grad is not None)
        critic2_grad_norm = sum(p.grad.norm().item() for p in self.critic_2.parameters() if p.grad is not None)

        critic1_clipped_grad_norm = T.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=1.0)
        critic2_clipped_grad_norm = T.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=1.0)

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

        actor_grad_norm = sum(p.grad.norm().item() for p in self.actor.parameters() if p.grad is not None)

        actor_clipped_grad_norm = T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
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

        # Update Replay Buffer priorities if using PER or N-Step-PER
        if self.buffer_type in [ 'per', 'n-step-per' ]:
            self.memory.update_priorities(indices, priorities)
            log_data.update({
                'per/avg_priority': np.mean(priorities),
                'per/max_priority': np.max(priorities),
                'hist:per/priority_distribution': priorities,
                'per/mean_weight': np.mean(weights.cpu().numpy()),
                'per/max_weight': np.max(weights.cpu().numpy()),
            })
        log_data.update({
            'buffer/length': len(self.memory)
        })

        log_data.update({
            'Losses/actor_loss': actor_loss.item(),
            'Losses/critic_1_loss': critic_1_loss.item(),
            'Losses/critic_2_loss': critic_2_loss.item(),
            'HyperParam/alpha': self.get_alpha().item(),
            'Metrics/entropy': -log_probs.mean().item(),
            'Metrics/q1_mean': q1_old.mean().item(),
            'Metrics/q1_std': q1_old.std().item(),
            'Metrics/q1_min': q1_old.min().item(),
            'Metrics/q1_max': q1_old.max().item(),
            'Metrics/critic1_grad_norm': critic1_grad_norm,
            'Metrics/critic2_grad_norm': critic2_grad_norm,
            'Metrics/actor_grad_norm': actor_grad_norm,
            'Metrics/q_target_mean': q_target.mean().item(),
            'Metrics/q_target_std': q_target.std().item()
        })
        return log_data
