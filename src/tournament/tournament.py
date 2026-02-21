import csv
from omegaconf import OmegaConf
from pathlib import Path

import numpy as np
import wandb
from openskill.models import PlackettLuce

from src.environments import SparseRewardHockeyEnv
from src.agent_factory import agent_factory
from src.named_agent import NamedAgent
from src.evaluation import Evaluator
from src.episode import Outcome
from src import wandb_utils


CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "tournament.yaml"

DEFAULT_GAUSS_SIGMA = 20.0
DEFAULT_GAUSS_SCALE = 0.5
DEFAULT_GAMES_PER_MATCH = 4

DEFAULT_ROUNDS = 10
DEFAULT_LOGGING_FREQUENCY = 2
DEFAULT_CSV_RANKING_PATH = Path("tournament_ranking.csv")


class RatingStorage:
    """
    Stores PlackettLuce ratings for agents and computes ranks.
    """

    DEFAULT_MU: float = 25.0
    DEFAULT_SIGMA: float = 25.0 / 3.0

    def __init__(self, agent_names: list[str]) -> None:
        self.model = PlackettLuce()
        self.ratings: dict[str, dict[str, float]] = {}
        for name in agent_names:
            self._set_rating(name, self.DEFAULT_MU, self.DEFAULT_SIGMA)
        self.ranks: dict[str, int] = {}
        self.update_ranks()

    def _set_rating(self, name: str, mu: float, sigma: float) -> None:
        """Set mu, sigma and score for an agent."""
        self.ratings[name] = {"mu": mu, "sigma": sigma, "score": mu - 3.0 * sigma}

    def update_ranks(self) -> None:
        """Resort agents by score and update ranks dict."""
        sorted_names = sorted(
            self.ratings.keys(),
            key=lambda n: self.ratings[n]["score"],
            reverse=True,
        )
        self.ranks = {name: pos for pos, name in enumerate(sorted_names)}

    def rank_position(self, name: str) -> int:
        """Returns agent's current rank position."""
        return self.ranks[name]

    def update_after_match(
        self,
        agent1_name: str,
        agent2_name: str,
        score1: float,
        score2: float,
    ) -> None:
        """Update PlackettLuce ratings after a match."""
        r1, r2 = self.ratings[agent1_name], self.ratings[agent2_name]

        [[p1], [p2]] = self.model.rate(
            [
                [self.model.create_rating([r1["mu"], r1["sigma"]], "player1")],
                [self.model.create_rating([r2["mu"], r2["sigma"]], "player2")],
            ],
            scores=[score1, score2],
        )

        self._set_rating(agent1_name, p1.mu, p1.sigma)
        self._set_rating(agent2_name, p2.mu, p2.sigma)

    def ranked_ratings(self) -> list[tuple[str, dict[str, float]]]:
        """Return (name, rating_dict) pairs sorted by score."""
        return sorted(
            self.ratings.items(),
            key=lambda item: item[1]["score"],
            reverse=True,
        )


class GaussLeaderboardRater:
    def __init__(
        self,
        rating_store: RatingStorage,
        scale: float = DEFAULT_GAUSS_SCALE,
        sigma: float = DEFAULT_GAUSS_SIGMA,
    ) -> None:
        self.store = rating_store
        self.scale = scale
        self.sigma = sigma

    def rate(self, name1: str, name2: str) -> float:
        pos1 = self.store.rank_position(name1)
        pos2 = self.store.rank_position(name2)
        signed_dist = pos1 - pos2
        return np.exp(-(signed_dist**2) / (2 * self.sigma**2)) * self.scale


class Matchmaker:
    def __init__(
        self,
        agents: dict[str, NamedAgent],
        rating_store: RatingStorage,
        games_per_match: int = DEFAULT_GAMES_PER_MATCH,
        gauss_scale: float = DEFAULT_GAUSS_SCALE,
        gauss_sigma: float = DEFAULT_GAUSS_SIGMA,
    ) -> None:
        self.agents = agents
        self.store = rating_store
        self.games_per_match = games_per_match

        self.rater = GaussLeaderboardRater(
            rating_store, scale=gauss_scale, sigma=gauss_sigma
        )

    def _find_pairs(self, agent_names: list[str]) -> list[tuple[str, str]]:
        rng = np.random.default_rng()
        paired = set()
        pairs: list[tuple[str, str]] = []

        for i, name1 in enumerate(agent_names):
            if name1 in paired:
                continue

            match_qualities = [
                (name2, self.rater.rate(name1, name2))
                for name2 in agent_names[i + 1 :]
                if name2 not in paired
            ]
            if not match_qualities:
                continue

            _, matched_scores = zip(*match_qualities, strict=True)
            quality_sum = sum(matched_scores)
            normalised = [q / quality_sum for q in matched_scores]

            match_idx = rng.choice(len(match_qualities), p=normalised)
            chosen_name = match_qualities[match_idx][0]

            if rng.choice([True, False]):
                pairs.append((name1, chosen_name))
            else:
                pairs.append((chosen_name, name1))

            paired.add(name1)
            paired.add(chosen_name)

        return pairs

    def run_round(self) -> None:
        pairs = self._find_pairs(list(self.agents.keys()))

        env = SparseRewardHockeyEnv()
        evaluator = Evaluator()

        for name1, name2 in pairs:
            agent1 = self.agents[name1]
            agent2 = self.agents[name2]

            wins1, wins2 = 0, 0
            for _ in range(self.games_per_match):
                episode, _ = evaluator.run_episode(env, agent1, agent2)
                wins1 += episode.outcome == Outcome.WIN
                wins2 += episode.outcome == Outcome.LOSS

            self.store.update_after_match(name1, name2, float(wins1), float(wins2))
        self.store.update_ranks()


def log_to_wandb_and_csv(
    rating_store: RatingStorage,
    round_idx: int,
    csv_path: Path,
) -> None:
    ranking = rating_store.ranked_ratings()
    wandb_payload: dict[str, float] = {}
    for name, r in ranking:
        wandb_payload[f"mu/{name}"] = r["mu"]
        wandb_payload[f"sigma/{name}"] = r["sigma"]
        wandb_payload[f"score/{name}"] = r["score"]
    wandb.log(wandb_payload, step=round_idx)

    with csv_path.open("w") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "agent", "mu", "sigma", "score"])
        for pos, (name, r) in enumerate(ranking, start=1):
            writer.writerow(
                [
                    round_idx,
                    pos,
                    name,
                    f"{r['mu']:.6f}",
                    f"{r['sigma']:.6f}",
                    f"{r['score']:.6f}",
                ]
            )


def create_agents(config: OmegaConf) -> dict[str, NamedAgent]:
    agents = {}
    for agent_name, agent_cfg in config.agents.items():
        agents[agent_name] = agent_factory(agent_name, agent_cfg)
    return agents


def main(config: OmegaConf) -> None:
    agents = create_agents(config)

    num_rounds: int = config.get("num_rounds", DEFAULT_ROUNDS)
    logging_freq: int = config.get(
        "logging_frequency_rounds", DEFAULT_LOGGING_FREQUENCY
    )
    wandb.init(
        entity=wandb_utils.TEAM_NAME,
        project=wandb_utils.PROJECT_NAME,
        name=config.get("run_name", "tournament"),
        config=OmegaConf.to_container(config, resolve=True),
    )
    csv_path = Path(config.get("csv_ranking_path", DEFAULT_CSV_RANKING_PATH))

    rating_storage = RatingStorage(list(agents.keys()))
    matchmaker = Matchmaker(
        agents=agents,
        rating_store=rating_storage,
    )

    for round_idx in range(1, num_rounds + 1):
        matchmaker.run_round()
        if round_idx % logging_freq == 0 or round_idx == num_rounds:
            log_to_wandb_and_csv(rating_storage, round_idx, csv_path)

    wandb.finish()


if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as f:
        cfg = OmegaConf.load(f)
        main(cfg)
