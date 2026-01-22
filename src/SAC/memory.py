import numpy as np
from collections import deque

class MemoryBuffer:
    def store_transition(self, state, action, reward, next_state, done):
        raise NotImplementedError

    def sample_buffer(self, batch_size):
        raise NotImplementedError

    def flush(self):
        pass

    def reset(self):
        pass

    def update_priorities(self, batch_indices, batch_priorities):
        pass

    def __len__(self):
        raise NotImplementedError


class ReplayBuffer(MemoryBuffer):
    """
    A fixed-size buffer to store transitions, implemented with PyTorch tensors.
    """

    def __init__(self, max_size, input_shape, n_actions, device = 'cpu'):
        """
        Initialize the replay buffer.

        Args:
            max_size (int): Maximum size of the buffer.
            input_shape (Tuple[int, ...]): Shape of the state observations.
            n_actions (int): Dimensionality of the action space.
            device (str): Device to store tensors ('cpu' or 'cuda').
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a single transition in the buffer.

        Args:
            state (torch.Tensor): The current state.
            action (torch.Tensor): The action taken.
            reward (float): The reward received after taking the action.
            next_state (torch.Tensor): The next state observed after the action.
            done (bool): Whether the episode ended.
        """
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1


    def sample_buffer(self, batch_size):
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): The number of samples to retrieve.

        Returns:
            Tuple[torch.Tensor]: A batch of (states, actions, rewards, next_states, terminals).
        """
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class PrioritizedReplayBuffer(MemoryBuffer):
    """
    Proportional Experience Prioritization
    """
    def __init__(self, capacity, alpha=0.6, beta_start = 0.4, beta_frames=100000):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1 #for beta calculation
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.mem_cntr = 0

    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def store_transition(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

        self.mem_cntr = min(1 + self.mem_cntr, self.capacity)

    def sample_buffer(self, batch_size):
        N = len(self.buffer)
        if N == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # calc P = p^a/sum(p^a)
        probs = prios ** self.alpha
        P = probs/probs.sum()

        #gets the indices depending on the probability p
        indices = np.random.choice(N, batch_size, p=P)
        samples = [self.buffer[idx] for idx in indices]

        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        #Compute importance-sampling weight
        weights = (N * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = abs(prio)

    def __len__(self):
        return len(self.buffer)

class NStepCollector(MemoryBuffer):
    """
    N-Step Experience Collector
    Collects n-step transitions and stores them in the provided replay buffer.
    """
    def __init__(self, n_step, gamma, replay_buffer):
        self.n_step = n_step
        self.gamma = gamma
        self.replay_buffer = replay_buffer
        self.n_step_buffer = deque(maxlen=n_step)
        self.mem_cntr = 0

    def store_transition(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) < self.n_step:
            return

        reward, next_state, done = self._get_n_step_info()
        state, action = self.n_step_buffer[0][:2]

        self.replay_buffer.store_transition(state, action, reward, next_state, done)
        self.mem_cntr = self.replay_buffer.mem_cntr

    def _get_n_step_info(self):
        reward, next_state, done = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[2], transition[3], transition[4]
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)

        return reward, next_state, done

    def flush(self):
        while len(self.n_step_buffer) > 0:
            reward, next_state, done = self._get_n_step_info()
            state, action = self.n_step_buffer[0][:2]
            self.replay_buffer.store_transition(state, action, reward, next_state, done)
            self.n_step_buffer.popleft()

    def reset(self):
        self.n_step_buffer.clear()

    def sample_buffer(self, batch_size):
        return self.replay_buffer.sample_buffer(batch_size)

    def update_priorities(self, batch_indices, batch_priorities):
        self.replay_buffer.update_priorities(batch_indices, batch_priorities)

    def __len__(self):
        return len(self.replay_buffer)
