from .noise import Noise
from .gaussian import GaussianNoise
from .ou_noise import OUNoise
from .pink_noise import PinkNoise
from .zero_noise import ZeroNoise


__all__ = [
    "Noise",
    "GaussianNoise",
    "OUNoise",
    "PinkNoise",
    "NoiseFactory",
    "ZeroNoise",
    "get_noise_types",
]

GAUSSIAN = 'gaussian'
OU       = 'ou'
PINK     = 'pink'
ZERO     = 'zero'

_NOISE_REGISTER = {
    GAUSSIAN: GaussianNoise,
    OU:       OUNoise,
    PINK:     PinkNoise,
    ZERO:     ZeroNoise
}

def get_noise_types():
    return list(_NOISE_REGISTER.keys())


class NoiseFactory:
    @staticmethod
    def get_noise(noise_string, *args, **kwargs):
        try:
            cls = _NOISE_REGISTER[noise_string.lower()]
        except:
            raise ValueError(f"Unknown noise type '{noise}'")
        return cls(*args, **kwargs)

