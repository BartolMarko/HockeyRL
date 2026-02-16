from __future__ import annotations

import os
import argparse
import uuid

from helper import create_agent_Nth_episode, get_resume_episode_number

from comprl.client import Agent, launch_client


class HockeyAgent(Agent):
    """A hockey agent that can be weak or strong."""

    def __init__(self, experiment_name, episode_number=-1) -> None:
        super().__init__()

        self.hockey_agent = create_agent_Nth_episode(
            experiment_name=experiment_name, n=episode_number,
            inference_only=True)

    def get_step(self, observation: list[float]) -> list[float]:
        action = self.hockey_agent.act(observation).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        if stats[0] == stats[1]:
            text_result = "tied"
        print(
            f"GAME END: {text_result} with score: {stats[0]} vs {stats[1]}"
        )


def initialize_agent(agent_args: list[str]) -> Agent:
    experiment_name = "sac-v4-pink-4-step-per-league-env-reset"

    if experiment_name is None:
        raise ValueError(
                "You must specify an experiment name using --experiment_name")

    exp_path = f"results/{experiment_name}/models"
    episode_number = get_resume_episode_number(exp_path)
    print(f"Loading the latest episode: {episode_number} for "
          f"experiment: {experiment_name}")

    agent = HockeyAgent(experiment_name=experiment_name,
                        episode_number=episode_number)

    # And finally return the agent.
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()
