import numpy as np

from .noise import Noise

class GaussianNoise(Noise):
    def __init__(self, action_dim):
        self.shape = action_dim

    def sample(self):
        return np.random.normal(0., 1., size=self.shape)