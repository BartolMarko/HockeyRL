import numpy as np

class ReplayBuffer:
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
