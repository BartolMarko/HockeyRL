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


class HockeyAgent(Agent):
    """A hockey agent that can be weak or strong."""

    def __init__(self, weak: bool) -> None:
        super().__init__()

        self.hockey_agent = h_env.BasicOpponent(weak=weak)

    def get_step(self, observation: list[float]) -> list[float]:
        # NOTE: If your agent is using discrete actions (0-7), you can use
        # HockeyEnv.discrete_to_continous_action to convert the action:
        #
        # from hockey.hockey_env import HockeyEnv
        # env = HockeyEnv()
        # continuous_action = env.discrete_to_continous_action(discrete_action)

        action = self.hockey_agent.act(observation).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


AGENT_PATH  = "../results5/td3_HockeyEnv_72000-s699.pth"
CONFIG_PATH = "./config.yaml"
class TD3Agent(Agent):
    def __init__(self):
        super().__init__()
        env = h_env.HockeyEnv()
        obs_space = env.observation_space
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        cfg = Config(CONFIG_PATH)
        self.model = TD3(obs_space, action_space, cfg['td3'])
        self.model.restore_state(torch.load(AGENT_PATH))
    
    def get_step(self, observation: list[float]) -> list[float]:
        action = self.model.act(observation).tolist()
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


# Function to initialize the agent.  This function is used with `launch_client` below,
# to lauch the client and connect to the server.
def initialize_agent(agent_args: list[str]) -> Agent:
    # Use argparse to parse the arguments given in `agent_args`.
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--path'
    # )
    # parser.add_argument(
    #     '--config'
    # )
    # parser.add_argument(
    #     "--agent",
    #     type=str,
    #     choices=["weak", "strong", "random"],
    #     default="weak",
    #     help="Which agent to use.",
    # )
    # args = parser.parse_args(agent_args)


    # # Initialize the agent based on the arguments.
    # agent: Agent
    # if args.agent == "weak":
    #     agent = HockeyAgent(weak=True)
    # elif args.agent == "strong":
    #     agent = HockeyAgent(weak=False)
    # elif args.agent == "random":
    #     agent = RandomAgent()
    # else:
    #     raise ValueError(f"Unknown agent: {args.agent}")


    # And finally return the agent.
    agent = TD3Agent()
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()