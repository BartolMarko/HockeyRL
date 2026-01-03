import numpy as np
from comprl.client.agent import Agent
from hockey import hockey_env as h_env

class NamedAgent(Agent):
    """
    Agent with a name attribute, should be used for evaluation purposes.
    
    Functions that should be overriden are on_start_game, on_end_game and get_step.
    """
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name


class WeakBot(NamedAgent):
    def __init__(self) -> None:
        super().__init__(name="WeakBot")
        self.bot = h_env.BasicOpponent(weak=True)
    
    def get_step(self, obs: np.ndarray) -> np.ndarray:
        return self.bot.act(obs)


class StrongBot(NamedAgent):
    def __init__(self) -> None:
        super().__init__(name="StrongBot")
        self.bot = h_env.BasicOpponent(weak=False)
    
    def get_step(self, obs: np.ndarray) -> np.ndarray:
        return self.bot.act(obs)