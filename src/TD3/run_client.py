from __future__ import annotations

import argparse
import uuid

import hockey.hockey_env as h_env
import numpy as np
from gymnasium import spaces
import torch

from src.TD3.config_reader import Config
from src.TD3.td3 import TD3


from comprl.client import Agent, launch_client


class RandomAgent(Agent):
    """A hockey agent that simply uses random actions."""

    def get_step(self, observation: list[float]) -> list[float]:
        return np.random.uniform(-1, 1, 4).tolist()

    def on_start_game(self, game_id) -> None:
        print("game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


AGENT_PATH_1  = "./models/td3/136k/checkpoint_step_136000_model.pt"
CONFIG_PATH_1 = "./models/td3/136k/checkpoint_step_136000_config.yaml"

AGENT_PATH_2 = "./models/td3/177k/checkpoint_step_177000_model.pt"
CONFIG_PATH_2 = "./models/td3/177k/checkpoint_step_177000_config.yaml"
class TD3Agent(Agent):
    def __init__(self):
        super().__init__()
        env = h_env.HockeyEnv()
        cfg1 = Config(CONFIG_PATH_1)
        cfg2 = Config(CONFIG_PATH_2)

        TD3.enhance_cfg(cfg1, env)
        TD3.enhance_cfg(cfg2, env)

        model1 = TD3(cfg1)
        model1.restore_state(torch.load(AGENT_PATH_1))

        model2 = TD3(cfg2)
        model2.restore_state(torch.load(AGENT_PATH_2))

        self.models = [model1, model2]
        self.current_agent = 0
    
    def get_step(self, observation: list[float]) -> list[float]:
        action = self.models[self.current_agent].act(observation).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id, byteorder="big"))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )

        lost = not result
        if (lost and stats[0] != stats[1]):
            self.current_agent = (self.current_agent + 1) % 2
            print("Switched to", self.current_agent)


def initialize_agent(agent_args: list[str]) -> Agent:
    agent = TD3Agent()
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()