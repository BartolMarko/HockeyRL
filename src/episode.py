import numpy as np
import torch
from enum import Enum

PUCK_X_COORDINATE_INDEX = 12
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
ACTION_INDICES_FOR_MIRRORING = [
    1,  # y force
    2,  # torque
]


class Possession(Enum):
    LEFT = "left"
    RIGHT = "right"


def get_puck_possession(obs: np.ndarray) -> Possession:
    """Determine which side has possession of the puck based on the observation."""
    puck_x_position = obs[PUCK_X_COORDINATE_INDEX]
    if puck_x_position < 0:
        return Possession.LEFT
    else:
        return Possession.RIGHT


class Outcome(Enum):
    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"


class Episode(object):
    """
    Storage object for a single episode.

    At each time step t, it stores:
    - obs[t]: observation at time step t
    - action[t]: action taken at time step t
    - opponent_action[t]: opponent's action at time step t
    - reward[t]: reward received after taking action at time step t
    - done: boolean flag indicating if the episode has terminated after the last added transition.
    - outcome: final outcome of the episode, not None if the episode has terminated.

    At the last time step T, obs[T] is stored but action[T], opponent_action[T], and reward[T] are not defined.
    """

    def __init__(
        self,
        init_obs: np.ndarray,
        device: str = "cuda",
        max_episode_length: int = 251,
        action_dim: int = 4,
        opponent_action_dim: int = 4,
    ):
        self.device = torch.device(device)
        dtype = torch.float32
        self.max_episode_length = max_episode_length
        self.obs = torch.empty(
            (max_episode_length + 1, *init_obs.shape), dtype=dtype, device=self.device
        )
        self.obs[0] = torch.tensor(init_obs, dtype=dtype, device=self.device)
        self.action = torch.empty(
            (max_episode_length, action_dim), dtype=torch.float32, device=self.device
        )
        self.opponent_action = torch.empty(
            (max_episode_length, opponent_action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.reward = torch.zeros(
            (max_episode_length,), dtype=torch.float32, device=self.device
        )
        self.done = False
        self.length = 0
        self.outcome = None
        self.first_puck_possession = get_puck_possession(init_obs)

    def __len__(self):
        """
        Return the number of transitions stored in the episode.
        Number of observations is length + 1.
        """
        return self.length
    
    def __getitem__(self, idx):
        if (not self.done) or (idx < self.length - 1):
            done = np.array(0., dtype=np.float32)
        else:
            done = np.array(1., dtype=np.float32)

        return (self.obs[idx].detach().cpu().numpy(), 
                self.action[idx].detach().cpu().numpy(),
                self.opponent_action[idx].detach().cpu().numpy(),
                self.reward[idx].detach().cpu().numpy(),
                done)

    @property
    def first(self):
        return len(self) == 0

    def __add__(
        self, transition: tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]
    ):
        """Add a transition to the episode."""
        self.add(*transition)
        return self

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        opponent_action: np.ndarray,
        reward: float,
        done: bool,
    ):
        assert not self.done, "Episode has terminated. Can not add more transitions."
        assert self.length < self.max_episode_length, "Episode buffer is full."

        self.obs[self.length + 1] = torch.tensor(
            obs, dtype=self.obs.dtype, device=self.obs.device
        )
        self.action[self.length] = torch.tensor(
            action, dtype=self.action.dtype, device=self.action.device
        )
        self.opponent_action[self.length] = torch.tensor(
            opponent_action,
            dtype=self.opponent_action.dtype,
            device=self.opponent_action.device,
        )
        self.reward[self.length] = reward
        self.done = done
        self.length += 1

        if done:
            self.obs = self.obs[: self.length + 1]
            self.action = self.action[: self.length]
            self.opponent_action = self.opponent_action[: self.length]
            self.reward = self.reward[: self.length]
            if reward > 0:
                self.outcome = Outcome.WIN
            elif reward < 0:
                self.outcome = Outcome.LOSS
            else:
                self.outcome = Outcome.DRAW

    def mirror(self) -> None:
        """Mirror the episode (up-down mirroring). Changes the episode in place."""
        for idx in OBSERVATION_INDICES_FOR_MIRRORING:
            self.obs[:, idx] *= -1

        for idx in ACTION_INDICES_FOR_MIRRORING:
            self.action[:, idx] *= -1
            self.opponent_action[:, idx] *= -1
