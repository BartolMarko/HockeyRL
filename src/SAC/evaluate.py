import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import helper
from hockey import hockey_env as h_env

from agent import Agent
import imageio

def evaluate(env, agent, opponent, num_episodes, step=None, render=False, save=None):
    """Evaluate a trained agent and optionally save a video."""
    win_count, lose_count, draw_count = 0, 0, 0
    frames = []
    for i in range(num_episodes):
        obs, _ = env.reset()
        obs_opponent = env.obs_agent_two()
        done = False
        while not done:
            if render or save is not None:
                mode = 'human' if render else 'rgb_array'
                frames.append(env.render(mode=mode))
            action = agent.plan(obs, eval_mode=True)
            if not isinstance(action, np.ndarray):
                action = action.cpu().numpy()
            opponent_action = opponent.get_step(obs_opponent)
            obs, reward, done, _, info = env.step(np.hstack([action, opponent_action]))
            obs_opponent = env.obs_agent_two()

        if reward < 0:
            lose_count += 1
        elif reward > 0:
            win_count += 1
        else:
            draw_count += 1
    opponent.record_play_scores(lose_count, win_count, draw_count)
    if save is not None:
        imageio.mimsave(save, frames, fps=30)
        print(f"Saved evaluation video to {save}.")

    return win_count, lose_count, draw_count

def evaluate_against_pool(env, agent, opponent_pool, num_episodes: int = 100, step: int = 0):
    """Evaluate a trained agent against a pool of opponents."""
    overall_stats = {}
    for opponent in opponent_pool.get_all_opponents():
        win_count, lose_count, draw_count = evaluate(env, agent, opponent, num_episodes, step)
        overall_stats[opponent.name] = {
            'win': win_count,
            'lose': lose_count,
            'draw': draw_count
        }
        print(f"Against {opponent.name}: Wins: {win_count}, Losses: {lose_count}, Draws: {draw_count}.")
    return overall_stats


def main():
    with open('config.yaml', 'r') as f:
        cfg = OmegaConf.load(f)

    env = h_env.HockeyEnv()
    cfg = helper.set_env_params(cfg, env)
    cfg.resume = True
    agent = Agent(cfg)

    BEST_EXPERIMENT_NAME = 'reward-v0-sac'
    cfg_2 = OmegaConf.load(Path('results') / BEST_EXPERIMENT_NAME / 'config.yaml')
    cfg_2 = helper.set_env_params(cfg_2, env)
    cfg_2.resume = True
    best_so_far = Agent(cfg_2)

    EVAL_EPISODES = 100
    cfg.eval_episodes = EVAL_EPISODES
    print(f"Evaluation over {EVAL_EPISODES} episodes each: ")
    opponents = {
            "WEAK": h_env.BasicOpponent(weak=True),
            "STRG": h_env.BasicOpponent(weak=False),
            "BEST": best_so_far
    }
    all_stats = {'win': 0, 'lose': 0, 'draw': 0}
    for opponent_name, opponent in opponents.items():
        win_count, lose_count, draw_count = evaluate(env, agent, opponent, EVAL_EPISODES, step=0, render=False)
        all_stats['win'] += win_count
        all_stats['lose'] += lose_count
        all_stats['draw'] += draw_count
        rates = (win_count / cfg.eval_episodes, lose_count / cfg.eval_episodes, draw_count / cfg.eval_episodes)
        print(f"{opponent_name}: Wins: {win_count}, Losses: {lose_count}, Draws: {draw_count}, Rates: {rates}.")

    print(f"Overall: Wins: {all_stats['win']}, Losses: {all_stats['lose']}, Draws: {all_stats['draw']}.")
    win_rate = all_stats['win'] / (len(opponents) * EVAL_EPISODES)
    lose_rate = all_stats['lose'] / (len(opponents) * EVAL_EPISODES)
    draw_rate = all_stats['draw'] / (len(opponents) * EVAL_EPISODES)
    rates = (win_rate, lose_rate, draw_rate)
    print("Rates: " + "{:.2f}, {:.2f}, {:.2f}".format(*rates))

if __name__ == '__main__':
    main()
