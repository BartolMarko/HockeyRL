from .noise import Noise
from .gaussian import GaussianNoise
from .ou_noise import OUNoise
from .pink_noise import PinkNoise


__all__ = [
    "Noise",
    "GaussianNoise",
    "OUNoise",
    "PinkNoise",
    "NoiseFactory"
    "get_noise_types",
]

GAUSSIAN = 'gaussian'
OU       = 'ou'
PINK     = 'pink'

_NOISE_REGISTER = {
    GAUSSIAN: GaussianNoise,
    OU:       OUNoise,
    PINK:     PinkNoise
}

def get_noise_types():
    return list(_NOISE_REGISTER.keys())


class NosieFactory:
    @staticmethod
    def get_noise(noise_string, *args, **kwargs):
        try:
            cls = _NOISE_REGISTER[noise_string.lower()]
        except:
            raise ValueError(f"Unknown noise type '{noise}'")
        return cls(*args, **kwargs)

