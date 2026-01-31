import itertools

import numpy as np
import torch
from torch import nn
import wandb
from pathlib import Path

from src.TD3.actor_critic import ActorCritic
from src.TD3.config_reader import Config
from src.TD3.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, PERNumpy, NStepRollOut
from src.named_agent import NamedAgent
from src.episode import Episode


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


MODEL_FILE      = "model.pt"
CHECKPOINT_FILE = "checkpoint_step_%d"
CONFIG_FILE     = "config.yaml"

class TD3(NamedAgent):
    def __init__(self, config : dict):
        super().__init__('TD3')
        # self.obs_space = config['observation_space']
        self._obs_dim = config['obs_dim']
        self._action_n = config['action_dim']
        # self.act_space = config['action_space']
        self._config = {
            "gamma": .99,
            "buffer_size": int(1e6),
            "batch_size": 128,
            "actor": {"hidden_sizes": [256, 256], 
                      "learning_rate": 1e-4},
            "critic": {"hidden_sizes": [256, 256],
                       "learning_rate": 1e-3},
            "polyak":.995,
            'target_noise': 0.2,
            'noise_clip':0.5,
            'policy_delay':2,
            'prioritized_replay': False,
            'pr_alpha': 0.6,
            'pr_beta': 0.4,
            "pr_epsilon": 1e-6
        } 

        self._config.update(config)

        self.pr_replay = self._config['prioritized_replay']
        self.polyak = self._config['polyak']

        # self._action_n = self.act_space.shape[0]
        # self._obs_dim = self.obs_space.shape[0]
        print(f"DEBUG grad clip norm: {self._config.get('grad_clip_norm')}")

        # self.high = torch.from_numpy(self.act_space.high).to(device)
        # self.low  = torch.from_numpy(self.act_space.low).to(device)
        self.high = torch.as_tensor(self._config['act_space_high']).to(device)
        self.low  = torch.as_tensor(self._config['act_space_low']).to(device)

        self.output_activation = lambda x : self.low + (self.high - self.low) * (nn.Tanh()(x) + 1.)/2

        self.model = ActorCritic(self._obs_dim, self._action_n, 
                                 actor_hidden_sizes=self._config['actor']['hidden_sizes'],
                                 actor_activation_fun=nn.ReLU(), 
                                 actor_ouput_activation_fun=self.output_activation,
                                 critic_hidden_sizes=self._config['actor']['hidden_sizes'],
                                 critic_activation_fun=nn.ReLU(),
                                 use_layernorm=self._config.get('use_layernorm', False)
                                 ).to(device)
        
        self.model_target = ActorCritic(self._obs_dim, self._action_n, 
                                 actor_hidden_sizes=self._config['critic']['hidden_sizes'],
                                 actor_activation_fun=nn.ReLU(), 
                                 actor_ouput_activation_fun=self.output_activation,
                                 critic_hidden_sizes=self._config['critic']['hidden_sizes'],
                                 critic_activation_fun=nn.ReLU(),
                                 use_layernorm=self._config.get('use_layernorm', False)
                                 ).to(device)
        
        self._hard_copy_nets()
        
        self.policy_optimizer = torch.optim.Adam(self.model.actor.parameters(), 
                                                 lr = self._config['actor']['learning_rate'])
        
        self.q_params = list(itertools.chain(self.model.critic1.parameters(), 
                                             self.model.critic2.parameters()))
        self.critic_optimizer = torch.optim.Adam(self.q_params,
                                                 lr = self._config['critic']['learning_rate'])
        
        self.N = self._config.get('rollout', 1)
        print("DEBUG, using rollout: ", self.N)
        self.rollout = NStepRollOut(self.N, self._config['gamma'])
        
        if self.pr_replay:
            self.buffer = PERNumpy(self._obs_dim, self._action_n, self._config['buffer_size'], 
                                   alpha = self._config['pr_alpha'], beta = self._config['pr_beta'])
        else:
            self.buffer = ReplayBuffer(self._obs_dim, self._action_n, self._config['buffer_size'])

    def _hard_copy_nets(self):
        self.model_target.restore_state(self.model.state())

    def _copy_nets(self):
        with torch.no_grad():
            for w, w_targ in zip(self.model.parameters(), self.model_target.parameters()):
                w_targ.data.mul_(self.polyak)
                w_targ.data.add_((1-self.polyak)*w.data)

    def store_transition(self, transition):
        self.rollout.add_transition(*transition)
        if self.rollout.is_full():
            self.buffer.add_transition(*self.rollout.pop())

    def store_episode(self, episode : Episode):
        for i in range(len(episode)):
            ob, act, _, rew, done = episode[i]
            ob_new = episode.obs[i + 1].detach().cpu().numpy()
            self.store_transition((ob, act, rew, ob_new, done))
        if done:
            self.on_episode_end()

    def compute_q_loss(self, data):
        if self.pr_replay:
            obs, act, rew, obs_new, done, weights, inds = data
        else:
            obs, act, rew, obs_new, done = data
            weights = torch.ones_like(rew, device=rew.device)

        q1 = self.model.q1(obs, act)
        q2 = self.model.q2(obs, act)

        with torch.no_grad():
            pi_targ = self.model_target.actor(obs_new)
            eps = torch.randn_like(pi_targ, device=device) * self._config['target_noise']
            eps = torch.clamp(eps, -self._config['noise_clip'], self._config['noise_clip'])

            a2 = pi_targ + eps
            a2 = torch.clamp(a2, self.low, self.high)

            q1_pi_targ = self.model_target.q1(obs_new, a2)
            q2_pi_targ = self.model_target.q2(obs_new, a2)

            q_pi_targ  = torch.min(q1_pi_targ, q2_pi_targ)
            target = rew + (self._config['gamma']**self.N)*(1-done)*q_pi_targ
        
        td_error1 = q1 - target
        td_error2 = q2 - target
        td_errors = torch.abs(td_error1 + td_error2) * .5

        loss_q1 = (weights * (td_error1*td_error1)).mean()
        loss_q2 = (weights * (td_error2*td_error2)).mean()
        
        loss_q = loss_q1 + loss_q2

        return loss_q, td_errors
    
    def compute_actor_loss(self, data):
        obs = data[0]
        action = self.model.actor(obs)
        q1_pi = self.model.q1(obs, action)
        return -q1_pi.mean()
    
    def update(self, t):

        data = self.buffer.sample_torch(self._config['batch_size'])

        self.critic_optimizer.zero_grad()
        loss_q, td_errors = self.compute_q_loss(data)
        loss_q.backward()

        if self._config.get('grad_clip_norm') is not None:
            nn.utils.clip_grad_norm_(self.q_params, max_norm=self._config['grad_clip_norm'])

        self.critic_optimizer.step()

        if self.pr_replay:
            # td_errs_npy = td_errors.detach().cpu().numpy()
            # new_pr = np.abs(td_errs_npy) + self._config['pr_epsilon']
            # new_pr = td_errors.detach().cpu().numpy() + self._config['pr_epsilon']
            new_pr = np.clip(td_errors.detach().cpu().numpy() + self._config['pr_epsilon'],
                             self._config['pr_epsilon'],
                             10.0)
            inds = data[-1]
            self.buffer.update_priorities(inds, new_pr)

        loss_pi = None

        if t % self._config['policy_delay'] == 0:

            self.policy_optimizer.zero_grad()
            loss_pi = self.compute_actor_loss(data)
            loss_pi.backward()
            if self._config.get('grad_clip_norm') is not None:
                nn.utils.clip_grad_norm_(self.model.actor.parameters(), max_norm=self._config['grad_clip_norm'])
            self.policy_optimizer.step()

            self._copy_nets()
    
        return loss_q.item(), loss_pi.item() if loss_pi is not None else None
    
    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
        return self.model.act(obs)
    
    def get_step(self, obs):
        return self.act(obs)

    def state(self):
        return self.model.state()
    
    def restore_state(self, state):
        self.model.restore_state(state)
        self._hard_copy_nets()

    def save_to_wandb(self, wandb_run, step):
        save_dir = Path(wandb_run.dir) / (CHECKPOINT_FILE % step)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state(), save_dir / MODEL_FILE)
        Config.save_as_yaml(self._config, str(save_dir / CONFIG_FILE))

        wandb.save(str(save_dir / MODEL_FILE))
        wandb.save(str(save_dir / CONFIG_FILE))

    def save_locally(self, directory, step):
        save_dir = Path(directory) / (CHECKPOINT_FILE % step)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state(), save_dir / MODEL_FILE)
        Config.save_as_yaml(self._config, str(save_dir / CONFIG_FILE))

    def reset_priorities(self):
        if self.pr_replay:
            self.buffer.reset_priorities()

    def on_episode_end(self):
        while self.rollout.can_pop():
            self.buffer.add_transition(*self.rollout.pop())
        #self.rollout.reset()

    def get_policy_config(self):
        return {
            "input_size": self._obs_dim,
            "hidden_sizes": self._config['actor']['hidden_sizes'],
            "output_size": self._action_n,
            "activation_func": nn.ReLU(),
            "output_activation": self.output_activation,
            "use_layernorm": self._config.get('use_layernorm', False),
            "state_dict": self.model.actor.state_dict()
        }
    
    @staticmethod
    def enhance_cfg(cfg, env):
        cfg['obs_dim']        = env.observation_space.shape[0]
        cfg['action_dim']     = env.action_space.shape[0] // 2
        cfg['act_space_high'] = env.action_space.high[:cfg['action_dim']].tolist()
        cfg['act_space_low']  = env.action_space.low[:cfg['action_dim']].tolist()

        
