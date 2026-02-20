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
    def get_noise(cfg, action_dim):
        match cfg['action_noise'].lower():
            case 'gaussian':
                return GaussianNoise(action_dim)
            case 'ou':
                return OUNoise(action_dim, theta=cfg.get('ou_theta', 0.15), dt=cfg.get('dt', 1.0))
            case 'pink':
                return PinkNoise(action_dim, f_min=cfg.get('pink_f_min', 0))
            case 'zero':
                return ZeroNoise(action_dim)
            case _:
                raise ValueError(f"Unknown noise type '{cfg['action_noise']}'")

