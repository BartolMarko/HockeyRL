import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import helper
from hockey import hockey_env as h_env
from agent import Agent
import opponents as opp
import imageio

def evaluate(env, agent, opponent, num_episodes, step=None, render=False, save=None, heatmap=False):
    """Evaluate a trained agent and optionally save a video."""
    win_count, lose_count, draw_count = 0, 0, 0
    if heatmap:
        hm = helper.HeatmapTracker()
        if step is None:
            step = 0
    frames = []
    for i in range(num_episodes):
        obs, _ = env.reset()
        obs_opponent = env.obs_agent_two()
        done = False
        while not done:
            if heatmap:
                hm.record_step(obs)
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
    if heatmap:
        hm.save_heatmap(f"heatmap_{agent.name}_vs_{opponent.name}_step{step}.png")
    return win_count, lose_count, draw_count

def evaluate_against_pool(env, agent, opponent_pool, num_episodes: int = 100, step: int | None = None, heatmap: bool = False):
    """Evaluate a trained agent against a pool of opponents."""
    overall_stats = {}
    for opponent in opponent_pool.get_all_opponents():
        win_count, lose_count, draw_count = evaluate(env, agent, opponent, num_episodes, step, heatmap=heatmap)
        overall_stats[opponent.name] = {
            'win': win_count,
            'lose': lose_count,
            'draw': draw_count
        }
        print(f"Against {opponent.name}: Wins: {win_count}, Losses: {lose_count}, Draws: {draw_count}.")
    return overall_stats

def main(args):
    if len(args) < 1:
        with open('config.yaml', 'r') as f:
            cfg = OmegaConf.load(f)
        experiment_name = cfg.exp_name
    else:
        experiment_name = args[0]

    env = h_env.HockeyEnv()
    agent = helper.load_agent_from_config(experiment_name, env)

    EVAL_EPISODES = 100
    print(f"Evaluation over {EVAL_EPISODES} episodes each: ")

    with open('best_agents.yaml', 'r') as f:
        opponents_cfg = OmegaConf.load(f)
    opponent_pool = opp.get_opponent_pool(opponents_cfg, env)

    all_stats = evaluate_against_pool(env, agent, opponent_pool, EVAL_EPISODES, heatmap=True)
    opponents = opponent_pool.get_all_opponents()
    strongest_opponent = max(opponents, key=lambda o: o.get_win_rate())
    print("\nSummary of Evaluation:")
    print("-" * 65)
    print(f"{'Experiment':45} | {'Wins':5} | {'Loss':4} | {'Draw':4}")
    print("-" * 65)
    for opponent in opponents:
        if opponent.name == agent.name:
            continue
        stats = all_stats[opponent.name]
        all_stats['win'] = all_stats.get('win', 0) + stats['win']
        all_stats['lose'] = all_stats.get('lose', 0) + stats['lose']
        all_stats['draw'] = all_stats.get('draw', 0) + stats['draw']
        if opponent == strongest_opponent:
            print(f"{opponent.name:42}(*) | {stats['win']:5} | {stats['lose']:4} | {stats['draw']:4}")
        else:
            print(f"{opponent.name:45} | {stats['win']:5} | {stats['lose']:4} | {stats['draw']:4}")
    print("-" * 65)
    print(f"Overall: Wins: {all_stats['win']}, Losses: {all_stats['lose']}, Draws: {all_stats['draw']}.")
    win_rate = all_stats['win'] / (len(opponents) * EVAL_EPISODES)
    lose_rate = all_stats['lose'] / (len(opponents) * EVAL_EPISODES)
    draw_rate = all_stats['draw'] / (len(opponents) * EVAL_EPISODES)
    rates = (win_rate, lose_rate, draw_rate)
    print("Rates: " + "{:.2f}, {:.2f}, {:.2f}".format(*rates))

if __name__ == '__main__':
    import sys
    arguments = sys.argv[1:]
    main(arguments)
