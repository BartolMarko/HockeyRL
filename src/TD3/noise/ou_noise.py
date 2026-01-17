import numpy as np

from .noise import Noise

class OUNoise(Noise):
    def __init__(self, action_dim, theta: float = 0.15, dt: float = 1.0):
        self._shape = action_dim
        self._theta = theta
        self._dt = dt
        self.noise_prev = np.zeros(self._shape)
        self.reset()

    def sample(self) -> np.ndarray:
        noise = (
            self.noise_prev
            + self._theta * ( - self.noise_prev) * self._dt
            + np.sqrt(self._dt) * np.random.normal(size=self._shape)
        )
        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        self.noise_prev = np.zeros(self._shape)