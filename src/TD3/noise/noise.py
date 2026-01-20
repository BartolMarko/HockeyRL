import numpy as np

from abc import ABC, abstractmethod


class Noise:
    
    @abstractmethod
    def sample(self) -> np.ndarray:
        ...

    def reset(self) -> None:
        pass