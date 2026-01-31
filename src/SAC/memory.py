import numpy as np
from collections import deque

OBSERVATION_INDICES_FOR_MIRRORING = [
    1,  # y pos player one
    2,  # angle player one
    4,  # y vel player one
    5,  # angular vel player one
    7,  # y pos player two
    8,  # angle player two
    10,  # y vel player two
    11,  # angular vel player two
    13,  # y pos puck
    15,  # y vel puck
]
obs_mask = np.zeros(16, dtype=bool)
obs_mask[OBSERVATION_INDICES_FOR_MIRRORING] = True
ACTION_INDICES_FOR_MIRRORING = [
    1,  # y force
    2,  # torque
]
action_mask = np.zeros(4, dtype=bool)
action_mask[ACTION_INDICES_FOR_MIRRORING] = True
def mirror_buffer(state_batch, action_batch, reward_batch, next_state_batch, done_batch, batch_indices=None):
    """
    Mirrors the observations and actions
    Ref: src.episode.Episode.mirror() (Bartol)
    """
    if batch_indices == None:
        batch_indices = np.arange(state_batch.shape[0])
    mask = np.zeros(state_batch.shape[0], dtype=bool)
    mask[batch_indices] = True
    mask_with_observations = mask[:, None] & obs_mask[None, :]
    mask_with_actions = mask[:, None] & action_mask[None, :]
    obs = state_batch.copy()
    next_obs = next_state_batch.copy()
    action = action_batch.copy()
    obs[mask_with_observations] *= -1
    next_obs[mask_with_observations] *= -1
    action[mask_with_actions] *= -1
    return obs, action, reward_batch, next_obs, done_batch

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

    def stats(self):
        print("Memory Buffer Size:", len(self))

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def set_store_upside_down(self, value):
        pass

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

        self.mem_cntr = min(self.mem_cntr + 1, self.mem_size)


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

    def __len__(self):
        return min(self.mem_cntr, self.mem_size)

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

class VecReplayBuffer(MemoryBuffer):
    """
    A vectorized replay buffer that manages multiple replay buffers for parallel environments.
    """
    def __init__(self, num_envs, max_size, input_shape, n_actions, device = 'cpu'):
        self.num_envs = num_envs
        self.mem_size = max_size
        self.store_upside_down = False

        self.mem_cntr = 0
        self.occupancy = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros((self.mem_size,), dtype=np.float32)
        self.terminal_memory = np.zeros((self.mem_size,), dtype=bool)

    def _store_transition(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch, mirror=True):
        if state_batch.shape[0] != self.num_envs:
            raise ValueError("Batch size must match number of environments, got {} and {}".format(state_batch.shape[0], self.num_envs))
        indices = np.arange(self.mem_cntr, self.mem_cntr + self.num_envs) % self.mem_size

        self.state_memory[indices] = state_batch
        self.new_state_memory[indices] = next_state_batch
        self.action_memory[indices] = action_batch
        self.reward_memory[indices] = reward_batch
        self.terminal_memory[indices] = done_batch

        self.mem_cntr = (self.mem_cntr + self.num_envs) % self.mem_size
        self.occupancy = min(self.occupancy + self.num_envs, self.mem_size)
        if self.store_upside_down and mirror:
            mirrored_states, mirrored_actions, mirrored_rewards, mirrored_next_states, mirrored_dones = mirror_buffer(
                state_batch,
                action_batch,
                reward_batch,
                next_state_batch,
                done_batch
            )
            self._store_transition(mirrored_states, mirrored_actions, mirrored_rewards, mirrored_next_states, mirrored_dones, mirror=False)

    def store_transition(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        self._store_transition(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def sample_buffer(self, batch_size):
        indices = np.random.randint(0, self.mem_cntr, size=batch_size)
        states = self.state_memory[indices]
        states_ = self.new_state_memory[indices]
        actions = self.action_memory[indices]
        rewards = self.reward_memory[indices]
        dones = self.terminal_memory[indices]
        return states, actions, rewards, states_, dones

    def __len__(self):
        return self.occupancy

    def save(self, filename):
        np.savez_compressed(
            filename if filename.endswith('.npz') else filename + '.npz',
            state_memory=self.state_memory,
            new_state_memory=self.new_state_memory,
            action_memory=self.action_memory,
            reward_memory=self.reward_memory,
            terminal_memory=self.terminal_memory,
            mem_cntr=self.mem_cntr,
            occupancy=self.occupancy
        )

    def load(self, filename):
        data = np.load(
            filename if filename.endswith('.npz') else filename + '.npz'
        )
        self.state_memory = data['state_memory']
        self.new_state_memory = data['new_state_memory']
        self.action_memory = data['action_memory']
        self.reward_memory = data['reward_memory']
        self.terminal_memory = data['terminal_memory']
        self.mem_cntr = data['mem_cntr'].item()
        self.occupancy = data['occupancy'].item()

    def set_store_upside_down(self, value):
        self.store_upside_down = value

class VecPrioritizedReplayBuffer(VecReplayBuffer):
    """
    A vectorized prioritized replay buffer for parallel environments.
    Lol, reimplementing all of this again, but guess what? i can use sum trees now
    """
    def __init__(self, num_envs, capacity, input_shape, n_actions, alpha=0.6, beta_start = 0.4, beta_frames=100000):
        super().__init__(num_envs, capacity, input_shape, n_actions)
        self.alpha = alpha
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1 #for beta calculation

        self.max_priority = 1.0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def _store_transition(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch, mirror=False):
        indices = np.arange(self.mem_cntr, self.mem_cntr + self.num_envs) % self.mem_size
        super()._store_transition(state_batch, action_batch, reward_batch, next_state_batch, done_batch, mirror=mirror)
        priorities = np.full(self.num_envs, self.max_priority, dtype=np.float32)
        self.update_priorities(indices, priorities)

    def store_transition(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        self._store_transition(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        if self.store_upside_down:
            mirrored_states, mirrored_actions, mirrored_rewards, mirrored_next_states, mirrored_dones = mirror_buffer(
                state_batch,
                action_batch,
                reward_batch,
                next_state_batch,
                done_batch
            )
            self._store_transition(mirrored_states, mirrored_actions, mirrored_rewards, mirrored_next_states, mirrored_dones)

    def update_priorities(self, batch_indices, batch_priorities):
        self.max_priority = max(self.max_priority, np.max(batch_priorities))
        priorities = ( np.abs(batch_priorities) + 1e-6 ) ** self.alpha
        self.priorities[batch_indices] = priorities

    def sample_buffer(self, batch_size):
        if len(self) == 0:
            raise ValueError("Cannot sample from an empty buffer")
        prios = self.priorities[:len(self)]
        total_priority = prios.sum()
        sampling_probabilities = prios / total_priority

        batch_indices = np.random.choice(len(self), batch_size, p=sampling_probabilities)

        states = self.state_memory[batch_indices]
        states_ = self.new_state_memory[batch_indices]
        actions = self.action_memory[batch_indices]
        rewards = self.reward_memory[batch_indices]
        dones = self.terminal_memory[batch_indices]

        # P(i) = p_i / total_priority
        # w = (N * P) ^ -beta; N = len(buffer)
        priorities = self.priorities[batch_indices]
        sampling_probabilities = priorities / total_priority
        N = len(self)
        self.frame += 1
        weights = (N * sampling_probabilities) ** (-self.beta_by_frame(self.frame))
        weights /= weights.max()
        return states, actions, rewards, states_, dones, batch_indices, weights

    def save(self, filename):
        super().save(filename)
        np.savez_compressed(
            filename + '_priorities.npz',
            max_priority=self.max_priority,
            priorities=self.priorities,
            frame=self.frame
        )

    def load(self, filename):
        super().load(filename)
        data = np.load(filename + '_priorities.npz')
        self.max_priority = data['max_priority'].item()
        self.priorities = data['priorities']
        self.frame = data['frame'].item()

class VecNStepPERBuffer(VecPrioritizedReplayBuffer):
    """
    A vectorized n-step prioritized experience replay buffer for parallel environments.
    well, lets stick to just PER for now
    """
    def __init__(self, num_envs, n_step, gamma, capacity, input_shape, n_actions, alpha=0.6, beta_start = 0.4, beta_frames=100000):
        super().__init__(num_envs, capacity, input_shape, n_actions, alpha, beta_start, beta_frames)
        self.n_step = n_step
        self.gamma = gamma
        self.num_envs = num_envs
        self.n_step_history = deque(maxlen=n_step)
        self.mirror_n_step_history = deque(maxlen=n_step)

    def store_transition(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        self.n_step_history.append((state_batch, action_batch, reward_batch, next_state_batch, done_batch))
        if self.store_upside_down:
            mirrored_states, mirrored_actions, mirrored_rewards, mirrored_next_states, mirrored_dones = mirror_buffer(
                state_batch,
                action_batch,
                reward_batch,
                next_state_batch,
                done_batch
            )
            self.mirror_n_step_history.append((mirrored_states, mirrored_actions, mirrored_rewards, mirrored_next_states, mirrored_dones))

        if len(self.n_step_history) < self.n_step:
            return

        reward_batch, next_state_batch, done_batch = self._get_n_step_info()
        state_batch, action_batch = self.n_step_history[0][:2]

        super()._store_transition(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

        if self.store_upside_down:
            reward_batch, next_state_batch, done_batch = self._get_mirror_n_step_info()
            state_batch, action_batch = self.mirror_n_step_history[0][:2]
            super()._store_transition(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def _get_mirror_n_step_info(self):
        reward_batch, next_state_batch, done_batch = self.mirror_n_step_history[-1][-3:]

        for transition in reversed(list(self.mirror_n_step_history)[:-1]):
            r_batch, n_s_batch, d_batch = transition[2], transition[3], transition[4]
            reward_batch = r_batch + self.gamma * reward_batch * (1 - d_batch)
            next_state_batch, done_batch = np.where(d_batch[:, None], n_s_batch, next_state_batch), np.where(d_batch, d_batch, done_batch)

        return reward_batch, next_state_batch, done_batch

    def _get_n_step_info(self):
        reward_batch, next_state_batch, done_batch = self.n_step_history[-1][-3:]

        for transition in reversed(list(self.n_step_history)[:-1]):
            r_batch, n_s_batch, d_batch = transition[2], transition[3], transition[4]
            reward_batch = r_batch + self.gamma * reward_batch * (1 - d_batch)
            next_state_batch, done_batch = np.where(d_batch[:, None], n_s_batch, next_state_batch), np.where(d_batch, d_batch, done_batch)

        return reward_batch, next_state_batch, done_batch

    def flush(self):
        while len(self.n_step_history) > 0:
            reward_batch, next_state_batch, done_batch = self._get_n_step_info()
            state_batch, action_batch = self.n_step_history[0][:2]
            super().store_transition(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
            self.n_step_history.popleft()

    # needless to call, but whatever
    # note: this does NOT save the queues
    def save(self, filename):
        super().save(filename)

    def load(self, filename):
        super().load(filename)

class VecHindsightExperienceReplayBuffer(MemoryBuffer):
    """
    A vectorized HER buffer for parallel environments.
    """
    def __init__(self, num_envs, max_size, input_shape, n_actions, her_k):
        raise NotImplementedError("HER buffer not implemented yet.")

def get_memory_buffer(cfg):
    num_envs = cfg.get('num_envs', 1)
    buffer_type = cfg.get('buffer_type', 'replay')
    if num_envs > 1:
        if buffer_type == 'replay':
            memory = VecReplayBuffer(
                num_envs,
                cfg.buffer_max_size,
                cfg.input_dims,
                cfg.n_actions
            )
        elif buffer_type == 'per':
            memory = VecPrioritizedReplayBuffer(
                num_envs,
                cfg.buffer_max_size,
                cfg.input_dims,
                cfg.n_actions,
                alpha=cfg.get('per_alpha', 0.6)
            )
        elif buffer_type == 'n-step-per':
            memory = VecNStepPERBuffer(
                num_envs,
                cfg.n_step_buffer_n,
                cfg.gamma,
                cfg.buffer_max_size,
                cfg.input_dims,
                cfg.n_actions,
                alpha=cfg.get('per_alpha', 0.6)
            )
        else:
            raise NotImplementedError("Only 'replay' buffer type is implemented for vectorized environments.")
    else:
        if buffer_type == 'per':
            alpha = cfg.get('per_alpha', 0.6)
            memory = PrioritizedReplayBuffer(cfg.buffer_max_size, alpha=alpha)
        elif buffer_type == 'n-step-per':
            alpha = cfg.get('per_alpha', 0.6)
            base_buffer = PrioritizedReplayBuffer(cfg.buffer_max_size, alpha=alpha)
            memory = NStepCollector(cfg.n_step_buffer_n, cfg.gamma, base_buffer)
        else:
            memory = ReplayBuffer(cfg.buffer_max_size, cfg.input_dims, cfg.n_actions)
    if cfg.get('store_upside_down', False):
        memory.set_store_upside_down(True)
    return memory

#### TESTS ####

def test_vec_per():
    # Ugh, this takes time
    print("VecPrioritizedReplayBuffer...")
    num_envs=2
    mem_size=4
    memory = VecPrioritizedReplayBuffer(num_envs=num_envs, capacity=mem_size, input_shape=(4,), n_actions=1, alpha=1.0)
    dummy_obs = np.zeros((num_envs, 4))
    for _ in range(mem_size):
        memory.store_transition(dummy_obs, np.zeros((num_envs, 1)), np.zeros((num_envs,)), dummy_obs, np.zeros((num_envs,), dtype=bool))
    assert len(memory) == mem_size
    print("  Initial fill test passed.")
    memory.store_transition(dummy_obs, np.zeros((num_envs, 1)), np.zeros((num_envs,)), dummy_obs, np.zeros((num_envs,), dtype=bool))
    assert len(memory) == mem_size
    print("  Wrapping test passed.")
    batch, actions, rewards, next_batch, dones, indices, weights = memory.sample_buffer(batch_size=4)
    assert batch.shape == (4, 4)
    print("  Sampling test passed.")

    num_envs = 2
    mem_size = 8
    memory = VecPrioritizedReplayBuffer(num_envs=num_envs, capacity=mem_size, input_shape=(4,), n_actions=1, alpha=1.0)
    dummy_obs = np.zeros((num_envs, 4))
    memory.store_transition(dummy_obs, np.zeros((num_envs, 1)), np.zeros((num_envs,)), dummy_obs, np.zeros((num_envs,), dtype=bool))
    # test priority assignment
    assert np.isclose(memory.priorities[0], 1.0)
    assert np.isclose(memory.priorities[1], 1.0)
    print("  Priority assignment test passed.")

    indices = np.array([0, 1])
    new_priorities = np.array([0.5, 0.2], dtype=np.float32)
    memory.update_priorities(indices, new_priorities)
    assert np.isclose(memory.priorities[0], 0.5)
    assert np.isclose(memory.priorities[1], 0.2)
    print("  Priority update test passed.")

    num_envs = 1
    mem_size = 16
    memory = VecPrioritizedReplayBuffer(num_envs=num_envs, capacity=mem_size, input_shape=(4,), n_actions=1, alpha=1.0, beta_start=1)
    dummy_obs = np.zeros((num_envs, 4))
    memory.store_transition(dummy_obs, np.zeros((num_envs, 1)), np.zeros((num_envs,)), dummy_obs, np.zeros((num_envs,), dtype=bool))
    memory.store_transition(dummy_obs, np.zeros((num_envs, 1)), np.zeros((num_envs,)), dummy_obs, np.zeros((num_envs,), dtype=bool))
    batch, actions, rewards, next_batch, dones, indices, weights = memory.sample_buffer(batch_size=10)
    idx_0 = np.array([0], dtype=np.int64)
    idx_1 = np.array([1], dtype=np.int64)
    prio_10 = np.array([10.0], dtype=np.float32)
    prio_20 = np.array([20.0], dtype=np.float32)
    memory.update_priorities(idx_0, prio_10)
    memory.update_priorities(idx_1, prio_10)
    assert np.isclose(weights, np.ones_like(weights)).all()
    memory.update_priorities(idx_0, prio_10)
    memory.update_priorities(idx_1, prio_20)
    batch, actions, rewards, next_batch, dones, indices, weights = memory.sample_buffer(batch_size=10)
    for i, idx in enumerate(indices):
        if idx == 0:
            assert np.isclose(weights[i], 1.0)
        elif idx == 1:
            assert np.isclose(weights[i], 0.5)
    # All the math I had to do for this, lol
    print("  Importance-sampling weights test passed.")

    # test when upside down is set, every transition stored increments by 2
    num_envs=2
    mem_size=4
    memory = VecPrioritizedReplayBuffer(num_envs=num_envs, capacity=mem_size, input_shape=(16,), n_actions=4, alpha=1.0)
    memory.set_store_upside_down(True)
    dummy_obs = np.zeros((num_envs, 16))
    for _ in range(mem_size // 2):
        memory.store_transition(dummy_obs, np.zeros((num_envs, 4)), np.zeros((num_envs,)), dummy_obs+1, np.zeros((num_envs,), dtype=bool))
    assert len(memory) == mem_size
    print("  Upside-down storing test passed.")

    print("PASS")

def test_vec_nstep():
    print("VecNStepPERBuffer...")
    num_envs=2
    mem_size=4
    n_step=3
    gamma=0.9
    memory = VecNStepPERBuffer(num_envs=num_envs, n_step=n_step, gamma=gamma, capacity=mem_size, input_shape=(4,), n_actions=1, alpha=1.0)
    dummy_obs = np.zeros((num_envs, 4))
    for _ in range(mem_size):
        memory.store_transition(dummy_obs, np.zeros((num_envs, 1)), np.ones((num_envs,)), dummy_obs, np.zeros((num_envs,), dtype=bool))
    assert len(memory) == mem_size
    print("  Initial fill test passed.")
    memory.store_transition(dummy_obs, np.zeros((num_envs, 1)), np.ones((num_envs,)), dummy_obs, np.zeros((num_envs,), dtype=bool))
    assert len(memory) == mem_size
    print("  Wrapping test passed.")
    batch, actions, rewards, next_batch, dones, indices, weights = memory.sample_buffer(batch_size=4)
    assert batch.shape == (4, 4)
    print("  Sampling test passed.")

    # 3-step reward test
    n_step = 3
    num_envs = 1
    mem_size = 10
    gamma = 0.9
    memory = VecNStepPERBuffer(num_envs=num_envs, n_step=n_step, gamma=gamma, capacity=mem_size, input_shape=(4,), n_actions=1, alpha=1.0)
    dummy_obs = np.zeros((num_envs, 4))
    memory.store_transition(dummy_obs, np.zeros((num_envs, 1)), np.array([1.0]), dummy_obs, np.array([0], dtype=bool))
    assert len(memory) == 0
    dummy_obs = np.random.rand(*(dummy_obs.shape))
    memory.store_transition(dummy_obs, np.zeros((num_envs, 1)), np.array([1.0]), dummy_obs, np.array([0], dtype=bool))
    assert len(memory) == 0
    dummy_obs = np.random.rand(*(dummy_obs.shape))
    memory.store_transition(dummy_obs, np.zeros((num_envs, 1)), np.array([1.0]), dummy_obs, np.array([0], dtype=bool))
    assert len(memory) == 1
    batch, actions, rewards, next_batch, dones, indices, weights = memory.sample_buffer(batch_size=1)
    assert all(batch[0] == 0.0)
    print("  N-step sample test passed.")
    expected_reward = 1.0 + 0.9 * 1.0 + 0.9**2 * 1.0
    assert np.isclose(rewards[0], expected_reward)
    print("  N-step reward calculation test passed.")

    # test when upside down is set, every transition stored increments by 2
    num_envs=2
    mem_size=4
    n_step=3
    gamma=0.9
    memory = VecNStepPERBuffer(num_envs=num_envs, n_step=n_step, gamma=gamma, capacity=mem_size, input_shape=(16,), n_actions=4, alpha=1.0)
    memory.set_store_upside_down(True)
    dummy_obs = np.zeros((num_envs, 16))
    for _ in range(n_step * mem_size // 2):
        memory.store_transition(dummy_obs, np.zeros((num_envs, 4)), np.zeros((num_envs,)), dummy_obs+1, np.zeros((num_envs,), dtype=bool))
    assert len(memory) == mem_size
    print("  Upside-down storing test passed.")

    # test mathematics of n-step mirroring
    num_envs = 1
    mem_size = 10
    n_step = 3
    gamma = 0.9
    memory = VecNStepPERBuffer(num_envs=num_envs, n_step=n_step, gamma=gamma, capacity=mem_size, input_shape=(16,), n_actions=4, alpha=1.0)
    memory.set_store_upside_down(True)
    dummy_obs = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]])
    memory.store_transition(dummy_obs, np.array([[0.5, -1.0, 0.0, 1.0]]), np.array([1.0]), dummy_obs+1, np.array([0], dtype=bool))
    dummy_obs = np.array([[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]])
    memory.store_transition(dummy_obs, np.array([[0.0, 1.0, -0.5, -1.0]]), np.array([1.0]), dummy_obs+1, np.array([0], dtype=bool))
    dummy_obs = np.array([[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]])
    memory.store_transition(dummy_obs, np.array([[-0.5, 0.0, 1.0, 0.5]]), np.array([1.0]), dummy_obs+1, np.array([0], dtype=bool))
    assert len(memory) == 2
    batch, actions, rewards, next_batch, dones, indices, weights = memory.sample_buffer(batch_size=2)
    expected_reward = 1.0 + 0.9 * 1.0 + 0.9**2 * 1.0
    for i in range(2):
        assert np.isclose(rewards[i], expected_reward)
    print("  N-step mirroring reward calculation test passed.")
    print("PASS")

def test_env_mirror():
    print("Environment mirroring...")
    state_batch = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]])
    action_batch = np.array([[0.5, -1.0, 0.0, 1.0]])
    reward_batch = np.array([1.0])
    next_state_batch = np.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]])
    done_batch = np.array([False])
    mirrored_state, mirrored_action, _, mirrored_next_state, _ = mirror_buffer(
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        done_batch
    )
    assert mirrored_state[0, 1] == -2.0
    assert mirrored_state[0, 2] == -3.0
    assert mirrored_action[0, 1] == 1.0
    assert mirrored_next_state[0, 1] == -3.0
    assert mirrored_next_state[0, 2] == -4.0
    print("  Mirroring test passed.")
    print("PASS")

if __name__ == "__main__":
    test_vec_per()
    test_vec_nstep()
    test_env_mirror()
