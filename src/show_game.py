import pickle
import time
import argparse

from hockey import hockey_env as h_env

GAME_FILE_PATH = None
SLEEP_TIME_BETWEEN_STEPS = 0.05


def show_game(env: h_env.HockeyEnv, observations: list):
    for obs in observations:
        env.set_state(obs)
        env.render()
        time.sleep(SLEEP_TIME_BETWEEN_STEPS)


def show_games(games_data: dict):
    env = h_env.HockeyEnv()
    print(f"{games_data['user_names'][0]} vs {games_data['user_names'][1]}")
    for game_data in games_data["rounds"]:
        show_game(env, game_data["observations"])


if __name__ == "__main__":
    if GAME_FILE_PATH is None:
        GAME_FILE_PATH = (
            argparse.ArgumentParser(description="Show a saved game.")
            .add_argument(
                "--game-file",
                type=str,
                help="Path to the game file to show.",
            )
            .parse_args()
            .game_file
        )
    with open(GAME_FILE_PATH, "rb") as f:
        results = pickle.load(f)
    show_games(results)
