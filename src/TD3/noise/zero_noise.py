from . import Noise

class ZeroNoise(Noise):
    def __init__(self, action_dim):
        pass

    def sample(self):
        return 0.0