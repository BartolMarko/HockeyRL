import wandb
from omegaconf import OmegaConf
from queue import Queue

from src.episode import Episode, Outcome
from src.evaluation import Heatmap

TEAM_NAME = "wayne-gradientzky"
PROJECT_NAME = "hockey-rl"


class TrainingMonitor:
    def __init__(
        self,
        run_name: str,
        config: OmegaConf,
        project_name=PROJECT_NAME,
        team_name=TEAM_NAME,
        per_opponent_metrics_window_size: int = 20,
    ):
        """Initialize a training monitor.

        Args:
            run_name: Name of the wandb run.
            config: Config of model being trained to log to wandb.
            project_name: Name of the wandb project. Defaults to PROJECT_NAME.
            team_name: Name of the wandb team/entity. Defaults to TEAM_NAME.
            per_opponent_metrics_window_size: Number of recent episodes to consider when
                computing per-opponent training metrics.
        """
        self.run = wandb.init(
            entity=team_name,
            project=project_name,
            name=run_name,
            config=config,
        )
        self.per_opponent_metrics_window_size = per_opponent_metrics_window_size
        self.opponent_outcome_counts = {}
        self.opponent_outcome_queues = {}
        self.opponent_episode_counts = {}

        self.total_heatmap = Heatmap(device="cuda")
        self.per_opponent_heatmaps = {}

    def log_training_episode(
        self,
        opponent_name: str,
        episode: Episode,
        step: int,
        episode_index: int,
    ) -> None:
        if opponent_name not in self.opponent_outcome_counts:
            self.opponent_outcome_counts[opponent_name] = {
                Outcome.WIN: 0,
                Outcome.LOSS: 0,
                Outcome.DRAW: 0,
            }
            self.opponent_outcome_queues[opponent_name] = Queue(
                maxsize=self.per_opponent_metrics_window_size
            )
            self.opponent_episode_counts[opponent_name] = 0
            self.per_opponent_heatmaps[opponent_name] = Heatmap(device="cuda")

        if self.opponent_outcome_queues[opponent_name].full():
            old_outcome = self.opponent_outcome_queues[opponent_name].get()
            self.opponent_outcome_counts[opponent_name][old_outcome] -= 1

        self.opponent_outcome_queues[opponent_name].put(episode.outcome)
        self.opponent_outcome_counts[opponent_name][episode.outcome] += 1
        self.opponent_episode_counts[opponent_name] += 1
        queue_size = self.opponent_outcome_queues[opponent_name].qsize()

        self.total_heatmap.add_episode(episode)
        self.per_opponent_heatmaps[opponent_name].add_episode(episode)

        outcome_rates = {
            f"train/{outcome.value}_rate/{opponent_name}": count / queue_size
            for outcome, count in self.opponent_outcome_counts[opponent_name].items()
        }
        self.run.log(
            {
                **outcome_rates,
                f"train/episodes_played/{opponent_name}": self.opponent_episode_counts[
                    opponent_name
                ],
                f"train/episode_length/{opponent_name}": len(episode),
                f"train/episode_reward/{opponent_name}": episode.reward.sum(),
                "train/episode_index": episode_index,
                "train/episode_reward": episode.reward.sum(),
                "train/episode_length": len(episode),
            },
            step=step,
        )

    def finish_training(self):
        self.total_heatmap.save_to_wandb(
            self.run,
            title="Train heatmaps against all opponents",
            suffix="/train/all_opponents",
        )
        for opponent_name, heatmap in self.per_opponent_heatmaps.items():
            heatmap.save_to_wandb(
                self.run,
                title=f"Train heatmaps against {opponent_name}",
                suffix=f"/train/{opponent_name}",
            )
        self.run.finish()
