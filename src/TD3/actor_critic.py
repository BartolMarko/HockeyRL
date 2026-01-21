import torch
from torch import nn

from src.TD3.feedforward import FeedForward


class Actor(FeedForward):
    def __init__(self, input_size, hidden_sizes, output_size, **kwargs):
        super().__init__(input_size, hidden_sizes, output_size, **kwargs)

    def act(self, obs):
        return self.predict(obs)
    
    def get_step(self, obs):
        return self.act(obs)


class Critic(FeedForward):
    def __init__(self, obs_dim, act_dim, hidden_sizes, **kwargs):
        super().__init__(input_size=obs_dim + act_dim, hidden_sizes = hidden_sizes,
                         output_size=1, **kwargs)
    
    def q(self, obs, act):
        q = self.forward(torch.cat([obs, act], dim=-1).float())
        return torch.squeeze(q, -1)
    
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, *, actor_hidden_sizes, 
                 actor_activation_fun, actor_ouput_activation_fun,
                 critic_hidden_sizes, critic_activation_fun, 
                 use_layernorm = False):
        super().__init__()

        self.actor = Actor(obs_dim, actor_hidden_sizes, act_dim,
                           activation_func = actor_activation_fun, 
                           output_activation = actor_ouput_activation_fun, 
                           use_layernorm = use_layernorm)
        
        self.critic1 = Critic(obs_dim, act_dim, critic_hidden_sizes, 
                              activation_func = critic_activation_fun, 
                              use_layernorm = use_layernorm)
        self.critic2 = Critic(obs_dim, act_dim, critic_hidden_sizes, 
                              activation_func = critic_activation_fun,
                              use_layernorm=use_layernorm)
        
    def q1(self, o, a):
        return self.critic1.q(o, a)

    def q2(self, o, a):
        return self.critic2.q(o, a)
        
    def state(self):
        return (self.actor.state_dict(), self.critic1.state_dict(), self.critic2.state_dict())
    
    def restore_state(self, state):
        self.actor.load_state_dict(state[0])
        self.critic1.load_state_dict(state[1])
        self.critic2.load_state_dict(state[2])

    def act(self, obs):
        with torch.no_grad():
            return self.actor(obs).cpu().numpy()