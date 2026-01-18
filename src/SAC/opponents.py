import random
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

class OpponentInPool(NamedAgent):
    def __init__(self, agent, index, priority) -> None:
        if hasattr(agent, "name"):
            name = agent.name
        else:
            name = f"{str(agent)}_Pool_{index}"
        super().__init__(name=name)
        self.agent = agent
        self.priority = priority
        self.sample_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0

    def get_step(self, obs: np.ndarray) -> np.ndarray:
        if hasattr(self.agent, "get_step"):
            return self.agent.get_step(obs)
        elif hasattr(self.agent, "act"):
            return self.agent.act(obs)
        elif hasattr(self.agent, "plan"):
            return self.agent.plan(obs)
        else:
            raise NotImplementedError("The base agent does not have a method to get actions.")

    def get_win_rate(self):
        games_played = self.get_games_played()
        if games_played == 0:
            return 0.0
        return self.win_count / games_played

    def record_play_scores(self, win_count, loss_count, draw_count):
        self.win_count = win_count
        self.loss_count = loss_count
        self.draw_count = draw_count

    def get_games_played(self):
        return self.win_count + self.loss_count + self.draw_count

class OpponentPool:
    def __init__(self):
        self.name = "UniformSampler-OpponentPool"
        self.opponents = []
        self.priority = []
        self.last_sampled_indices = []

    def add_opponent(self, opponent, priority: float = 1.0):
        new_oppendent = OpponentInPool(opponent, len(self.opponents), priority)
        self.opponents.append(new_oppendent)
        self.priority.append(priority)

    def _sample_opponent_indices(self, index: int | None = None, count: int = 1):
        if len(self.opponents) == 1:
            return [0] * count
        count = max(min(count, len(self.opponents)), 1)
        total_priority = np.sum(self.priority)
        probabilities = [p / total_priority for p in self.priority]
        sample_indices = np.random.choice(len(self.opponents), size=count, p=probabilities)
        return sample_indices

    def sample_opponent(self, index: int | None = None, count: int = 1):
        sample_indices = self._sample_opponent_indices(index=index, count=count)
        self.last_sampled_indices = sample_indices
        sampled_opponents = [self.opponents[i] for i in sample_indices]
        for opponent in sampled_opponents:
            opponent.sample_count += 1
        if count == 1:
            return sampled_opponents[0]
        return sampled_opponents

    def update_opponent_priority(self, index: int, new_priority: float):
        if 0 <= index < len(self.opponents):
            self.priority[index] = new_priority
        else:
            raise IndexError("Opponent index out of range.")

    def update_last_sampled_opponent_priorities(self, new_priority: float | list[float]):
        if isinstance(new_priority, list):
            if len(new_priority) != len(self.last_sampled_indices):
                raise ValueError("Length of new_priority list must match number of last sampled opponents.")
            for i, idx in enumerate(self.last_sampled_indices):
                self.update_opponent_priority(i, new_priority[i])
        else:
            for idx in self.last_sampled_indices:
                self.update_opponent_priority(idx, new_priority)

    def get_last_opponents(self):
        return [self.opponents[i] for i in self.last_sampled_indices]

    def get_all_opponents(self):
        return self.opponents

    def show_scoreboard(self):
        for opponent in self.opponents:
            games_played = opponent.get_games_played()
            win_rate = (opponent.win_count / games_played * 100) if games_played > 0 else 0.0
            print(f"Opponent: {opponent.name}, Priority: {opponent.priority:.2f}, "
                  f"Games Played: {games_played}, Wins: {opponent.win_count}, "
                  f"Losses: {opponent.loss_count}, Draws: {opponent.draw_count}, "
                  f"Win Rate: {win_rate:.2f}%")

    def __len__(self):
        return len(self.opponents)

    def __add__(self, other):
        if not isinstance(other, OpponentPool):
            raise ValueError("Can only add another OpponentPool.")
        new_pool = OpponentPool()
        for opponent, priority in zip(self.opponents + other.opponents,
                                      self.priority + other.priority):
            new_pool.add_opponent(opponent.agent, priority)
        return new_pool

def get_bot_pool(cfg) -> OpponentPool:
    weak_prior = cfg.get('weak_bot_priority', 0.5)
    strg_prior = cfg.get('strg_bot_priority', 0.5)
    pool = OpponentPool()
    pool.add_opponent(WeakBot(), priority=weak_prior)
    pool.add_opponent(StrongBot(), priority=strg_prior)
    return pool

def get_opponent_pool(cfg) -> OpponentPool:
    pool = OpponentPool()
    opponents_cfg = cfg.get('opponents', [])
    for opp_cfg in opponents_cfg:
        opp_type = opp_cfg.get('type', 'WeakBot')
        priority = opp_cfg.get('priority', 1.0)
        if opp_type == 'WeakBot':
            opponent = WeakBot()
        elif opp_type == 'StrongBot':
            opponent = StrongBot()
        else:
            raise ValueError(f"Unknown opponent type: {opp_type}")
        pool.add_opponent(opponent, priority=priority)
    return pool
