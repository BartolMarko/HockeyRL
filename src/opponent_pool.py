import numpy as np
from queue import Queue

from src.named_agent import NamedAgent
from src.episode import Outcome

DEFAULT_PRIOR = {Outcome.WIN: 1, Outcome.LOSS: 1, Outcome.DRAW: 1}


class OpponentPool:
    def __init__():
        pass

    def add_opponent(self, opponent: NamedAgent):
        raise NotImplementedError("Should be implemented in child classes!")

    def add_episode_outcome(self, opponent: NamedAgent, outcome: Outcome):
        raise NotImplementedError("Should be implemented in child classes!")

    def sample_opponent(self) -> NamedAgent:
        raise NotImplementedError("Should be implemented in child classes!")

    def get_opponents(self) -> list[NamedAgent]:
        raise NotImplementedError("Should be implemented in child classes!")


class OpponentPoolThompsonSampling(OpponentPool):
    """
    Opponent pool using windowed Thompson sampling.
    For each opponent estimates the (P_win, P_draw, P_loss) by sampling from dirichlet distribution
    and chooses opponent with highest sampled P_loss + draw_weight * P_draw.
    Probabilities are estimated with respect to only the last window_size_episodes episodes.
    """

    def __init__(
        self,
        opponents: list[NamedAgent] = [],
        window_size_episodes: int = 200,
        prior: dict[Outcome, int] = DEFAULT_PRIOR,
        draw_weight=0.5,
    ) -> None:
        """Initialize the opponent pool with given opponents, window size, and same prior for each opponent."""
        self.opponents = []
        self.outcome_counts = {}
        self.window_size_episodes = window_size_episodes
        self.outcome_queue = Queue(maxsize=window_size_episodes)
        self.draw_weight = draw_weight

        for opponent in opponents:
            self.add_opponent(opponent, prior)

    def add_opponent(
        self, opponent: NamedAgent, prior: dict[Outcome, int] = DEFAULT_PRIOR
    ) -> None:
        """Add a new opponent to the pool with given prior outcome counts."""
        self.opponents.append(opponent)
        self.outcome_counts[opponent.name] = prior.copy()

    def add_episode_outcome(self, opponent: NamedAgent, outcome: Outcome) -> None:
        """Add the outcome of an episode against the given opponent to the pool statistics."""
        if self.outcome_queue.full():
            old_opponent_name, old_outcome = self.outcome_queue.get()
            self.outcome_counts[old_opponent_name][old_outcome] -= 1

        self.outcome_queue.put((opponent.name, outcome))
        self.outcome_counts[opponent.name][outcome] += 1

    def sample_opponent(self) -> NamedAgent:
        """Return the opponent with highest estimated (sampled) P_loss + draw_weight * P_draw."""
        sampled_scores = []
        for opponent in self.opponents:
            counts = self.outcome_counts[opponent.name]

            win_count = counts[Outcome.WIN]
            draw_count = counts[Outcome.DRAW]
            loss_count = counts[Outcome.LOSS]

            sampled_probabilities = np.random.dirichlet(
                np.array([win_count, draw_count, loss_count])
            )
            sampled_scores.append(
                sampled_probabilities[2] + self.draw_weight * sampled_probabilities[1]
            )

        chosen_index = int(np.argmax(sampled_scores))
        return self.opponents[chosen_index]

    def get_opponents(self) -> list[NamedAgent]:
        """Return the list of opponents in the pool."""
        return self.opponents


class OpponentPoolUniform(OpponentPool):
    """
    Opponent pool using uniform sampling.
    """

    def __init__(
        self,
        opponents: list[NamedAgent] = [],
    ) -> None:
        self.opponents = []
        for opponent in opponents:
            self.add_opponent(opponent)

    def add_opponent(self, opponent: NamedAgent) -> None:
        self.opponents.append(opponent)

    def add_episode_outcome(self, opponent: NamedAgent, outcome: Outcome) -> None:
        pass

    def sample_opponent(self) -> NamedAgent:
        return np.random.choice(self.opponents)

    def get_opponents(self) -> list[NamedAgent]:
        return self.opponents


def opponent_pool_factory(
    pool_type: str,
    opponents: list[NamedAgent],
    **kwargs,
) -> OpponentPool:
    """Factory function to create opponent pool instances based on the type."""
    match pool_type:
        case "ThompsonSampling":
            return OpponentPoolThompsonSampling(opponents, **kwargs)
        case "Uniform":
            return OpponentPoolUniform(opponents)
        case _:
            raise ValueError(f"Unknown opponent pool type: {pool_type}")
