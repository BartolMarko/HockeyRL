import random

import numpy as np
import torch

from src.TD3.algos import SumSegmentTree, MinSegmentTree
from src.TDMPC.helper import ReplayBuffer as PER

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, max_size : int = int(1e6)):
        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act = np.zeros((max_size, act_dim), dtype=np.float32)
        self.obs_new = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.rew = np.zeros(max_size, dtype=np.float32)
        self.done = np.zeros(max_size, dtype=np.float32)
        self.idx, self.size, self.max_size = 0, 0, max_size

    def add_transition(self, ob, act, rew, ob_new, done):
        self.obs[self.idx] = ob
        self.act[self.idx] = act 
        self.rew[self.idx] = rew
        self.obs_new[self.idx] = ob_new
        self.done[self.idx] = done
        self.idx = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        inds = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[inds],
            self.act[inds],
            self.rew[inds],
            self.obs_new[inds],
            self.done[inds]
        )
    
    def sample_torch(self, batch_size):
        batch = self.sample(batch_size)
        return tuple(torch.tensor(t, dtype=torch.float32).to(device) for t in batch)
    

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, obs_dim, act_dim, size, alpha, beta):
        super().__init__(obs_dim, act_dim, size)
        self.alpha = alpha
        self.beta  = beta

        it_capcaity = 1
        while it_capcaity < self.max_size:
            it_capcaity *= 2
        
        self.sum_tree = SumSegmentTree(it_capcaity)
        self.min_tree = MinSegmentTree(it_capcaity)
        self.max_priority = 1.0
    
    def add_transition(self, ob, act, rew, ob_new, done):
        idx = self.idx
        super().add_transition(ob, act, rew, ob_new, done)
        self.sum_tree[idx] = self.max_priority ** self.alpha
        self.min_tree[idx] = self.max_priority ** self.alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self.sum_tree.sum(0, self.size - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx  = self.sum_tree.find_prefixsum_idx(mass)
            res.append(idx)
        return res
    
    def _encode_sample(self, inds):
        return (
            self.obs[inds],
            self.act[inds],
            self.rew[inds],
            self.obs_new[inds],
            self.done[inds]
        )
    
    def sample(self, batch_size):
        inds = self._sample_proportional(batch_size)
        weights = []
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self.size) ** (-self.beta)

        for idx in inds:
            p_sample = self.sum_tree[idx] / self.sum_tree.sum()
            weight = (p_sample * self.size) ** (-self.beta)
            weights.append(weight / max_weight)

        weights = np.array(weights)
        encoded_sample = self._encode_sample(inds)
        return tuple(list(encoded_sample) + [weights, inds])

    def sample_torch(self, batch_size):
        batch = self.sample(batch_size)
        ret = []
        for i in range(len(batch) - 1):
            ret.append(torch.tensor(batch[i], dtype=torch.float32).to(device))
        ret.append(batch[-1])
        return ret
    
    def update_priorities(self, inds, priorities):
        assert len(inds) == len(priorities)

        for ind, priority in zip(inds, priorities):
            assert priority > 0, f"priority at {ind} is <=0, {priority}"
            assert 0 <= ind < self.size

            self.sum_tree[ind] = priority ** self.alpha
            self.min_tree[ind] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)


class PERNumpy:
    def __init__(self, obs_dim, act_dim, size, alpha, beta, max_size = int(1e6)):
        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act = np.zeros((max_size, act_dim), dtype=np.float32)
        self.obs_new = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.rew = np.zeros(max_size, dtype=np.float32)
        self.done = np.zeros(max_size, dtype=np.float32)
        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.alpha = alpha
        self.beta  = beta
        self.idx, self.size, self.max_size = 0, 0, max_size
        self.max_priority = 1.0

    def add_transition(self, ob, act, rew, ob_new, done):
        self.obs[self.idx] = ob
        self.act[self.idx] = act 
        self.rew[self.idx] = rew
        self.obs_new[self.idx] = ob_new
        self.done[self.idx] = done
        
        self.priorities[self.idx] = self.max_priority

        self.idx = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        valid_priorities = self.priorities[:self.size] ** self.alpha
        probs = valid_priorities / valid_priorities.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probs)
        # indices = torch.multinomial(probs, batch_size, replacement=True)

        # p_min = valid_priorities.min() / valid_priorities.sum()
        # max_weight = (p_min * self.size) ** (-self.beta)
        
        p_sample = probs[indices]
        weights = (p_sample * self.size) ** (-self.beta)
        weights = weights / weights.max()
        
        obs = self.obs[indices]
        act = self.act[indices]
        rew = self.rew[indices]
        obs_new = self.obs_new[indices]
        done = self.done[indices]
        
        return obs, act, rew, obs_new, done, weights, indices
    
    def sample_torch(self, batch_size):
        batch = self.sample(batch_size)
        ret = []
        for i in range(len(batch) - 1):
            ret.append(torch.tensor(batch[i], dtype=torch.float32).to(device))
        ret.append(batch[-1])
        return ret
    
    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max().item())
    