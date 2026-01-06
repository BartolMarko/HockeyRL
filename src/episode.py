import numpy as np
import torch
from enum import StrEnum


class Outcome(StrEnum):
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

    def __len__(self):
        """
        Return the number of transitions stored in the episode.
        Number of observations is length + 1.
        """
        return self.length

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
