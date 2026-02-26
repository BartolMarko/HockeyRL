from __future__ import annotations

import yaml
import signal
import uuid

from .helper import create_agent_Nth_episode, get_resume_episode_number

from comprl.client import Agent, launch_client


def load_config(config_path="src/SAC/comprl.yaml") -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


class HockeyAgent(Agent):
    """A hockey agent that can be weak or strong."""

    def __init__(self) -> None:
        super().__init__()

        self.cfg = load_config()

        self.hockey_agent = None
        self._create_agent()

        signal.signal(signal.SIGUSR1, self._update_agent)

    def current_time(self):
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _update_agent(self, signum, frame):
        print("Received signal to update the agent.")
        self.cfg = load_config()
        self._create_agent()

    def _create_agent(self):
        self.experiment_name = self.cfg['experiment_name']
        self.episode_number = self.cfg.get('episode_number', -1)
        exp_path = f"src/SAC/results/{self.experiment_name}/models"
        if self.hockey_agent is not None:
            print("Agent already exists. Overwriting it.")
        if self.episode_number == -1 or self.hockey_agent is not None:
            episode_number = get_resume_episode_number(exp_path)
            assert episode_number != 0, f"No episode found in {exp_path}"
        else:
            episode_number = self.episode_number
        print(f"Loading the latest episode: {episode_number} for "
              f"experiment: {self.experiment_name}")
        self.hockey_agent = create_agent_Nth_episode(
            experiment_name=self.experiment_name, n=episode_number,
            inference_only=True)

    def get_step(self, observation: list[float]) -> list[float]:
        action = self.hockey_agent.act(observation).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id}) at {self.current_time()}")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        if stats[0] == stats[1]:
            text_result = "tied"
        print(
            f"GAME END: {text_result} with score: {stats[0]} vs {stats[1]} at {self.current_time()}"
        )


def initialize_agent(agent_args: list[str]) -> Agent:
    # experiment_name = "sac-v4-pink-4-step-per-league-env-reset"

    agent = HockeyAgent()

    # And finally return the agent.
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()
