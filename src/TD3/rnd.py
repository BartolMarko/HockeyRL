import torch 
from torch import nn
import numpy as np

from src.TD3.feedforward import FeedForward

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
sources used:

https://github.com/jcwleo/random-network-distillation-pytorch/blob/master/utils.py
'''
class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean   = np.zeros(shape)
        self.var    = np.zeros(shape)
        self.count  = epsilon
    
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var  = np.var(x, axis=0)
        batch_count = len(x)

        delta = batch_mean - self.mean
        count = self.count + batch_count

        self.mean += delta * batch_count/count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / count
        self.var = M2 / count

        self.count = count

class RewardForwardFilter:
    '''this is for discounting intrinsic rewards!'''
    def __init__(self, gamma):
        self.gamma = gamma
        self.rewems = None
    
    def reset(self):
        self.rewems = None
    
    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems        


class RND:
    def __init__(self, obs_dim, hidden_sizes, output_size, gamma, lr=1e-4, beta=1.0, 
                 max_episode_length = 251):
        self.target     = FeedForward(obs_dim, hidden_sizes, output_size).to(device)
        self.pred       = FeedForward(obs_dim, hidden_sizes, output_size).to(device)
        self.beta       = beta

        self.obs_buffer     = np.empty((max_episode_length, obs_dim))
        self.int_rew_buffer = np.empty((max_episode_length,)) 
        self.ep_size        =0
        self.int_return_rms = RunningMeanStd()
        self.obs_rms        = RunningMeanStd(shape=obs_dim)
        self.forward_filter = RewardForwardFilter(gamma=gamma)

        self.optimizer  = torch.optim.Adam(
            self.pred.parameters(), lr=lr
        )

        for targ_p in self.target.parameters():
            targ_p.requires_grad = False

    def normalize_obs(self, obs):
        return np.clip(
            (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8),
            -5, 5
        )

    def get_intrinsic_reward(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs).to(device)

        with torch.no_grad():
            target = self.target(obs)
            pred   = self.pred(obs)
            err = (target - pred).pow(2).mean(-1)
        return err
    
    def compute_intrinsic_reward(self, obs):
        obs_normalized = self.normalize_obs(obs)
        obs_tensor = torch.as_tensor(obs_normalized).to(device)

        with torch.no_grad():
            target_features = self.target(obs_tensor)
            predictor_features = self.predictor(obs_tensor)
        
        int_reward = ((target_features - predictor_features) ** 2).mean().item()
        self.obs_buffer[self.ep_size] = obs
        self.int_rew_buffer[self.ep_size] = int_reward
        self.ep_size += 1
        
        return int_reward
    

    def on_episode_end(self):
        self.obs_rms.update(self.obs_buffer[:self.ep_size])
        self.obs_count += self.ep_size

        obs_normalized = self.normalize_obs(self.obs_buffer[:self.ep_size])
        obs_tensor = torch.as_tensor(obs_normalized).to(device)

        with torch.no_grad():
            t_out = self.target(obs_tensor)
        p_out = self.pred(obs_tensor)
        diff = t_out - p_out
        loss = diff.pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        int_rews = self._normalize_int_rewards(self.int_rew_buffer[:self.ep_size])
        return int_rews

    def _normalize_int_rewards(self, int_rewards):
        discounted_returns = []
        for rew in int_rewards:
            disc_ret = self.forward_filter.update(rew)
            discounted_returns.append(disc_ret)
        
        self.int_return_rms.update(np.array(discounted_returns))
        
        normalized_rewards = int_rewards / np.sqrt(self.int_return_rms.var)
        return normalized_rewards

    def update(self, obs):
        with torch.no_grad():
            t_out = self.target(obs)
        p_out = self.pred(obs)
        diff = (t_out- p_out)
        loss = (diff).pow(2).mean(dim=-1).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def state(self):
        return (self.pred.state_dict(), self.target.state_dict())
    
    def restore_state(self, state):
        self.pred.load_state_dict(state[0])
        self.target.load_state_dict(state[1])
        
        