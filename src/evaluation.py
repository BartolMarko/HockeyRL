import cv2
import numpy as np
import json
import torch
import wandb
import matplotlib.pyplot as plt
from typing import Optional
from hockey import hockey_env as h_env
from pathlib import Path

from src.named_agent import StrongBot, WeakBot
from src.episode import Episode, Outcome, Possession
from src.named_agent import NamedAgent, SACLastYearAgent
from src.agent_factory import create_td3_agent
from src.environments import DefenseModeImprovedEnv

from src.TDMPC.agent import TDMPCAgent


class VideoBuilder:
    def __init__(self, fps=30, header_height=60):
        self.fps = fps
        self.header_height = header_height

        self.frame_chunks = []

        self.width = None
        self.height = None

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1.0
        self.font_thickness = 2
        self.text_color = (0, 0, 0)
        self.bg_color = (255, 255, 255)

    def add_rgb_frames(self, frames_array: np.ndarray, title: str):
        """
        Add a chunk of RGB frames with a header title.
        Excpects frames_array of shape (num_frames, H, W, 3).
        """
        _, img_h, img_w, _ = frames_array.shape
        expected_total_h = img_h + self.header_height

        if self.width is None:
            self.width = img_w
            self.height = expected_total_h
        else:
            if img_w != self.width or expected_total_h != self.height:
                raise ValueError("Dimension mismatch with previous frames.")

        frames = frames_array.astype(np.uint8)
        pad_width = ((0, 0), (self.header_height, 0), (0, 0), (0, 0))
        padded_frames = np.pad(frames, pad_width, mode="constant", constant_values=255)

        header_block = np.full((self.header_height, self.width, 3), 255, dtype=np.uint8)

        (text_w, text_h), _ = cv2.getTextSize(
            title, self.font, self.font_scale, self.font_thickness
        )
        text_x = (self.width - text_w) // 2
        text_y = (self.header_height + text_h) // 2

        cv2.putText(
            header_block,
            title,
            (text_x, text_y),
            self.font,
            self.font_scale,
            self.text_color,
            self.font_thickness,
            cv2.LINE_AA,
        )
        padded_frames[:, : self.header_height, :, :] = header_block
        self.frame_chunks.append(padded_frames)

    def save_to_wandb(
        self, wandb_run: wandb.Run, video_label: str, step: Optional[int] = None
    ) -> None:
        """Saves video to wandb with given label. If no frames were added, does nothing."""
        if not len(self.frame_chunks):
            return

        merged_video_numpy = np.concatenate(self.frame_chunks, axis=0)
        merged_video_numpy = np.transpose(
            merged_video_numpy, axes=(0, 3, 1, 2)
        )  # (F, C, H, W)
        wandb_video = wandb.Video(merged_video_numpy, fps=self.fps, format="mp4")
        wandb_run.log({video_label: wandb_video}, step=step)

    def save(self, filename: str):
        """Saves video to mp4 file. If no frames were added, does nothing."""
        if not len(self.frame_chunks):
            return
        if not filename.lower().endswith(".mp4"):
            filename += ".mp4"

        print(f"Saving video to {filename}... ", end="")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(filename, fourcc, self.fps, (self.width, self.height))

        for chunk in self.frame_chunks:
            for i in range(chunk.shape[0]):
                bgr_frame = cv2.cvtColor(chunk[i], cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)

        out.release()
        print("Saved!")


class Heatmap:
    def __init__(
        self,
        device: str = "cpu",
        num_bins_h=32,
        num_bins_w=40,
        height=h_env.H,
        width=h_env.W,
    ):
        self.device = device
        self.height = height
        self.width = width
        self.num_bins_h = num_bins_h
        self.num_bins_w = num_bins_w

        self.x_bins = torch.linspace(
            -self.width / 2, self.width / 2, num_bins_w + 1, device=self.device
        )
        self.y_bins = torch.linspace(
            -self.height / 2, self.height / 2, num_bins_h + 1, device=self.device
        )

        self.players_heatmap = torch.zeros(
            (num_bins_h, num_bins_w), dtype=torch.float32, device=self.device
        )
        self.puck_heatmap = torch.zeros(
            (num_bins_h, num_bins_w), dtype=torch.float32, device=self.device
        )
        self.num_samples = 0

    def __add__(self, heatmap: "Heatmap") -> "Heatmap":
        """Add another heatmap to this one."""
        self.players_heatmap += heatmap.players_heatmap
        self.puck_heatmap += heatmap.puck_heatmap
        self.num_samples += heatmap.num_samples
        return self

    def add_episode(self, episode: Episode) -> None:
        """Add positions from the episode to the heatmaps."""
        player_positions = episode.obs[:, :2].to(self.device)
        opponent_positions = episode.obs[:, 6:8].to(self.device)
        puck_positions = episode.obs[:, 12:14].to(self.device)

        self._add_positions_to_heatmap(player_positions, self.players_heatmap)
        self._add_positions_to_heatmap(
            opponent_positions, self.players_heatmap, value=-1.0
        )
        self._add_positions_to_heatmap(puck_positions, self.puck_heatmap)
        self.num_samples += (
            len(episode) + 1
        )  # Length of episode is number of transitions

    def save_to_wandb(
        self,
        wandb_run: wandb.Run,
        title: str,
        step: Optional[int] = None,
        suffix: str = "",
    ) -> None:
        """Save heatmaps to wandb."""
        # TODO: Possible bug when saving heatmaps to wandb?
        try:
            fake_suffix = "temp"
            self.save(wandb_run.dir, title=title, suffix=fake_suffix)
            heatmap_image_path = Path(wandb_run.dir) / f"heatmaps{fake_suffix}.png"
            wandb_run.log(
                {f"heatmaps{suffix}": wandb.Image(heatmap_image_path)}, step=step
            )
        except:  # noqa: E722
            print(f"Failed to save heatmaps {title} to wandb.")

    def save(self, path: str | Path, title: str, suffix: str = "") -> None:
        """
        Save heatmaps as .npy files and as a combined png image.
        File names will be "players_heatmap{suffix}.npy", "puck_heatmap{suffix}.npy",
        and "heatmaps{suffix}.png".

        Args:
            path (str | Path): Directory to save the files in.
            title (str): Title for the heatmap image.
            suffix (str, optional): Suffix to append to file names.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        players_heatmap_np = (self.players_heatmap / self.num_samples).cpu().numpy()
        puck_heatmap_np = (self.puck_heatmap / self.num_samples).cpu().numpy()

        np.save(path / f"players_heatmap{suffix}.npy", players_heatmap_np)
        np.save(path / f"puck_heatmap{suffix}.npy", puck_heatmap_np)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(title)

        im1 = axes[0].imshow(
            players_heatmap_np,
            cmap="seismic",
            origin="lower",
            vmin=-np.max(np.abs(players_heatmap_np)),
            vmax=np.max(np.abs(players_heatmap_np)),
        )
        axes[0].set_title("Players Heatmap")
        axes[0].set_xticklabels([])
        axes[0].set_yticklabels([])
        plt.colorbar(im1, ax=axes[0], label="Player Visits")

        im2 = axes[1].imshow(puck_heatmap_np, cmap="Greens", origin="lower")
        axes[1].set_title("Puck Heatmap")
        axes[1].set_xticklabels([])
        axes[1].set_yticklabels([])
        plt.colorbar(im2, ax=axes[1], label="Puck Visits")

        plt.tight_layout()
        plt.savefig(path / f"heatmaps{suffix}.png")
        plt.close()

    def _add_positions_to_heatmap(
        self, positions: torch.Tensor, heatmap: torch.Tensor, value: float = 1.0
    ) -> None:
        x_indices = torch.bucketize(positions[:, 0].contiguous(), self.x_bins) - 1
        y_indices = torch.bucketize(positions[:, 1].contiguous(), self.y_bins) - 1

        valid_mask = (
            (x_indices >= 0)
            & (x_indices < self.num_bins_w)
            & (y_indices >= 0)
            & (y_indices < self.num_bins_h)
        )

        for x_idx, y_idx in zip(x_indices[valid_mask], y_indices[valid_mask]):
            heatmap[y_idx, x_idx] += value


class Evaluator:
    OVERALL_NAME = "overall"

    def __init__(self, device: str = "cpu"):
        self.device = device

    def run_episode(
        self,
        env: h_env.HockeyEnv,
        agent: NamedAgent,
        opponent: NamedAgent,
        render_mode: Optional[str] = None,
        every_n_steps: int = 1,
    ) -> tuple[Episode, Optional[np.ndarray]]:
        """
        Run a single episode between the agent and opponent in the given environment.
        Returns the episode object and optionally the video frames if render_mode is 'rgb_array'.
        If render_mode is 'human', the environment will be rendered to the screen.
        If render_mode is neither, no rendering is performed and video=None is returned.
        When every_n_steps > 1, steps are saved in episode only every_n_steps steps.
        (used for action repeat)
        """
        obs, _ = env.reset()
        obs_opponent = env.obs_agent_two()

        episode = Episode(obs, device=self.device)
        agent.on_start_game(game_id=None)
        opponent.on_start_game(game_id=None)

        episode_video = None if render_mode != "rgb_array" else []
        if render_mode == "rgb_array":
            episode_video.append(env.render(mode=render_mode))
        elif render_mode == "human":
            env.render(mode=render_mode)

        while not episode.done:
            for i in range(every_n_steps):
                action = agent.get_step(obs)
                opponent_action = opponent.get_step(obs_opponent)
                obs, reward, done, _, info = env.step(
                    np.hstack([action, opponent_action])
                )
                obs_opponent = env.obs_agent_two()

                if render_mode == "rgb_array":
                    episode_video.append(env.render(mode=render_mode))
                elif render_mode == "human":
                    env.render(mode=render_mode)
                if done:
                    break

            episode.add(
                obs=obs,
                action=action,
                opponent_action=opponent_action,
                reward=reward,
                done=done,
            )

        agent.on_end_game(result=None, stats=None)
        opponent.on_end_game(result=None, stats=None)

        episode_video = np.stack(episode_video) if episode_video is not None else None
        return episode, episode_video

    def evaluate_agent_and_save_metrics(
        self,
        env: h_env.HockeyEnv,
        agent: NamedAgent,
        opponents: list[NamedAgent],
        num_episodes: int,
        render_mode: Optional[str] = None,
        save_path: Optional[str | Path] = None,
        wandb_run: Optional[wandb.Run] = None,
        save_heatmaps: bool = False,
        save_episodes_per_outcome: dict[Outcome, int] | int = {
            Outcome.WIN: 0,
            Outcome.LOSS: 0,
            Outcome.DRAW: 0,
        },
        train_step: Optional[int] = None,
    ) -> dict[str, dict[str, int]]:
        """
        Evaluate the given agent against a list of opponents in the specified environment.

        For each opponent, runs num_episodes episodes and records the outcomes (win/loss/draw).
        Computes win/loss/draw rates per opponent and per first possession.
        If save_path is provided, saves the results and heatmaps to that directory (per opponent).
        If render_mode is 'rgb_array', saves videos of a specified number of episodes per outcome.
        If render_mode is 'human', renders the episodes to the screen.

        Args:
            env: HockeyEnv or wrapped HockeyEnv.
            agent: NamedAgent to evaluate.
            opponents: List of NamedAgents to evaluate against.
            num_episodes: Number of episodes to run per opponent.
            render_mode: 'human', 'rgb_array', or None. Defaults to None.
            save_path: Directory to save results and videos. Defaults to None.
            wandb_run: wandb Run object as target for logging. If None, no logging to wandb. Defaults to None.
            save_heatmaps: Whether to save heatmaps. Defaults to False.
            save_episodes_per_outcome: Number of episodes to save per outcome. If int, same number for all outcomes.
                Defaults to { Outcome.WIN: 0, Outcome.LOSS: 0, Outcome.DRAW: 0}.
            train_step: Current training step for logging purposes. Defaults to None.

        Returns:
            A dictionary mapping opponent names to their evaluation results (win/loss/draw counts and rates).
        """
        if isinstance(save_episodes_per_outcome, int):
            temp = save_episodes_per_outcome
            save_episodes_per_outcome = {outcome: temp for outcome in Outcome}

        results = {}
        for opponent in opponents:
            print(f"Evaluating {agent.name} against opponent: {opponent.name}")

            results_opponent = {
                side_name: {outcome: 0 for outcome in Outcome}
                for side_name in [
                    Possession.LEFT.value,
                    Possession.RIGHT.value,
                    self.OVERALL_NAME,
                ]
            }
            outcome_video_builders = {outcome: VideoBuilder() for outcome in Outcome}
            heatmap = Heatmap(device=self.device) if save_heatmaps else None

            for _ in range(num_episodes):
                episode, episode_video = self.run_episode(
                    env, agent, opponent, render_mode=render_mode
                )

                results_opponent[self.OVERALL_NAME][episode.outcome] += 1
                results_opponent[episode.first_puck_possession.value][
                    episode.outcome
                ] += 1
                if heatmap is not None:
                    heatmap.add_episode(episode)

                if (
                    render_mode == "rgb_array"
                    and results_opponent[self.OVERALL_NAME][episode.outcome]
                    <= save_episodes_per_outcome[episode.outcome]
                ):
                    outcome_video_builders[episode.outcome].add_rgb_frames(
                        episode_video,
                        title=f"{agent.name} vs {opponent.name}",
                    )

            results[opponent.name] = self._prettify_results(results_opponent)
            print(f"Results against {opponent.name}: {results_opponent}")
            if save_path is not None:
                self._save_per_opponent_metrics_locally(
                    save_path,
                    agent.name,
                    opponent.name,
                    results[opponent.name],
                    outcome_video_builders,
                    heatmap if save_heatmaps else None,
                )

            if wandb_run is not None:
                self._log_per_opponent_metrics_to_wandb(
                    wandb_run,
                    agent.name,
                    opponent.name,
                    results[opponent.name],
                    outcome_video_builders,
                    heatmap if save_heatmaps else None,
                    train_step,
                )

        if save_path is not None:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / "overall_results.json", "w") as f:
                json.dump(results, f, indent=4)

        return results

    @staticmethod
    def _log_per_opponent_metrics_to_wandb(
        wandb_run: wandb.Run,
        agent_name: str,
        opponent_name: str,
        results_opponent: dict[str, int],
        outcome_video_builders: dict[Outcome, VideoBuilder],
        heatmap: Optional[Heatmap] = None,
        train_step: Optional[int] = None,
    ) -> None:
        """Log the results for a single opponent to wandb."""
        print(f"Logging results against {opponent_name} to wandb...")
        wandb_run.log(
            {
                f"eval/{metric_name}/{opponent_name}": metric
                for metric_name, metric in results_opponent.items()
            },
            step=train_step,
        )

        for outcome, builder in outcome_video_builders.items():
            builder.save_to_wandb(
                wandb_run,
                video_label=f"eval/{agent_name}_vs_{opponent_name}_{outcome.value}",
                step=train_step,
            )

        if heatmap is not None:
            heatmap.save_to_wandb(
                wandb_run,
                title=f"{agent_name} vs {opponent_name} evaluation heatmaps",
                suffix=f"_{agent_name}_vs_{opponent_name}",
                step=train_step,
            )

    @staticmethod
    def _save_per_opponent_metrics_locally(
        save_path: str | Path,
        agent_name: str,
        opponent_name: str,
        results_opponent: dict[Outcome, int],
        outcome_video_builders: dict[Outcome, VideoBuilder],
        heatmap: Optional[Heatmap] = None,
    ) -> None:
        """Save the results for a single opponent to a local JSON file."""
        opponent_save_path = Path(save_path) / f"vs_{opponent_name}"
        opponent_save_path.mkdir(parents=True, exist_ok=True)

        with open(opponent_save_path / "results.json", "w") as f:
            json.dump(results_opponent, f, indent=4)

        for outcome, builder in outcome_video_builders.items():
            video_filename = (
                opponent_save_path / f"{opponent_name}_{outcome.value}_episode.mp4"
            )
            builder.save(str(video_filename))

        num_episodes = sum(results_opponent.values())
        if heatmap is not None:
            heatmap.save(
                opponent_save_path,
                title=f"{agent_name} vs {opponent_name} evaluation heatmaps ({num_episodes} episodes)",
                suffix=f"_{agent_name}_vs_{opponent_name}",
            )

    @classmethod
    def _prettify_results(
        cls, opponent_results: dict[str, dict[Outcome, int]]
    ) -> dict[str, float]:
        """Convert outcome counts to a dictionary with rates per opponent and first possession."""
        result = {
            f"{outcome.value}": count
            for outcome, count in opponent_results[cls.OVERALL_NAME].items()
        }
        result.update(
            {
                f"{outcome.value}_rate": count
                / sum(opponent_results[cls.OVERALL_NAME].values())
                for outcome, count in opponent_results[cls.OVERALL_NAME].items()
            }
        )
        result.update(
            {
                f"{possession}_possession_{outcome.value}": count
                for possession, possession_results in opponent_results.items()
                if possession != cls.OVERALL_NAME
                for outcome, count in possession_results.items()
            }
        )
        result.update(
            {
                f"{possession}_possession_{outcome.value}_rate": count
                / sum(opponent_results[possession].values())
                for possession, possession_results in opponent_results.items()
                if possession != cls.OVERALL_NAME
                and sum(opponent_results[possession].values()) > 0
                for outcome, count in possession_results.items()
            }
        )
        return result


if __name__ == "__main__":
    MODELS_PATH = Path(__file__).resolve().parent.parent / "models"
    TD3_FAKER_PATH = MODELS_PATH / "td3_benchmarks" / "td3_faker"
    TD3_BANK_SHOT_PATH = MODELS_PATH / "td3_benchmarks" / "td3_bank_shot"
    TD3_IMPROVED_PATH = MODELS_PATH / "td3_benchmarks" / "td3_improved"

    TRAINING_PATH_OLD = MODELS_PATH / "tdmpc2_mirror"
    CHECKPOINT_FINAL_OLD = TRAINING_PATH_OLD / "final"
    # CHECKPOINT_1_85M_PATH = TRAINING_PATH / "checkpoint_1_85m"
    # CHECKPOINT_1_5M_PATH = TRAINING_PATH / "checkpoint_1_5m"
    # CHECKPOINT_1_1M_PATH = TRAINING_PATH / "checkpoint_1_1m"
    # CHECKPOINT_800K_PATH = TRAINING_PATH / "checkpoint_800k"
    # CHECKPOINT_600K_PATH = TRAINING_PATH / "checkpoint_600k"
    # CHECKPOINT_400K_PATH = TRAINING_PATH / "checkpoint_400k"
    TRAINING_PATH = MODELS_PATH / "tdmpc2_action_repeat"
    CHECKPOINT_600K_PATH = TRAINING_PATH / "checkpoint_600k"
    CHECKPOINT_100K_PATH = TRAINING_PATH / "checkpoint_100k"
    env = DefenseModeImprovedEnv()
    tdmpc_agent = TDMPCAgent(
        CHECKPOINT_600K_PATH, tdmpc=None, step=600_000, eval_mode=True
    )
    checkpoint_100k = TDMPCAgent(
        CHECKPOINT_100K_PATH, tdmpc=None, step=100_000, name_suffix="_self_100k"
    )
    tdmpc_old = TDMPCAgent(
        CHECKPOINT_FINAL_OLD, tdmpc=None, step=2_000_000, name_suffix="_old"
    )
    # checkpoint_400k = TDMPCAgent(
    #     CHECKPOINT_400K_PATH, tdmpc=None, step=400_000, name_suffix="_self_400k"
    # )
    # checkpoint_600k = TDMPCAgent(
    #     CHECKPOINT_600K_PATH, tdmpc=None, step=600_000, name_suffix="_self_600k"
    # )
    strong_bot = StrongBot()
    weak_bot = WeakBot()
    evaluator = Evaluator(device="cuda")

    td3_faker = create_td3_agent(
        name="TD3_Faker",
        weights_path=TD3_FAKER_PATH / "model.pt",
        config_path=TD3_FAKER_PATH / "config.yaml",
    )
    td3_improved = create_td3_agent(
        name="TD3_Improved",
        weights_path=TD3_IMPROVED_PATH / "model.pt",
        config_path=TD3_IMPROVED_PATH / "config.yaml",
    )

    sac_last_year_agent = SACLastYearAgent(
        env=env,
    )
    results = evaluator.evaluate_agent_and_save_metrics(
        env=env,
        agent=tdmpc_agent,
        opponents=[strong_bot],
        num_episodes=30,
        render_mode="human",
        save_path=None,
        wandb_run=None,
        save_heatmaps=True,
        save_episodes_per_outcome={Outcome.WIN: 20, Outcome.LOSS: 50, Outcome.DRAW: 20},
        train_step=100_000,
    )
    print(results)
    # wandb_run.finish()
