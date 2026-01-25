import numpy as np
from omegaconf import OmegaConf
import time
import helper
from hockey import hockey_env as h_env
import puffer_wrapper as pfw

def puffer_evaluate(env, agent, opponent, num_episodes, step=None, render=False, save=None, heatmap=False):
    start_time = time.time()
    win_counts = np.zeros(env.num_envs, dtype=int)
    lose_counts = np.zeros(env.num_envs, dtype=int)
    draw_counts = np.zeros(env.num_envs, dtype=int)
    episodes_per_env = num_episodes // env.num_envs

    episodes_done = np.zeros(env.num_envs, dtype=int)
    obs_batch, _ = env.reset()
    obs_opponent_batch = env.obs_agent_two()
    done_batch = np.zeros(env.num_envs, dtype=bool)

    active_envs = np.ones(env.num_envs, dtype=bool)

    done_count = 0

    while done_count < num_episodes:
        agent_actions = agent.plan_batch(obs_batch, eval_mode=True)
        opponent_actions = opponent.get_step(obs_opponent_batch)
        combined_actions = np.hstack([agent_actions, opponent_actions])

        # IMP: Ensure float32 and clip to valid range
        combined_actions = np.clip(combined_actions, -1, 1).astype(np.float32)

        obs_batch, reward_batch, done_batch, _, info_batch = env.step(combined_actions)
        obs_opponent_batch = env.obs_agent_two()

        active_envs = (episodes_done < episodes_per_env)
        done_batch = np.logical_and(done_batch, active_envs)
        win_counts += (done_batch & (reward_batch > 0)).astype(int)
        lose_counts += (done_batch & (reward_batch < 0)).astype(int)
        draw_counts += (done_batch & (reward_batch == 0)).astype(int)

        episodes_done += done_batch.astype(int)
        done_count += np.sum(done_batch)

    total_wins = np.sum(win_counts)
    total_losses = np.sum(lose_counts)
    total_draws = np.sum(draw_counts)
    end_time = time.time()
    opponent.record_play_scores(total_losses, total_wins, total_draws)
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds.")
    return total_wins, total_losses, total_draws

def puffer_evaluate_against_pool(env, agent, opponent_pool, num_episodes: int = 100, step: int | None = None, heatmap: bool = False):
    overall_stats = {}
    players = opponent_pool.get_playable_opponents()
    print("Starting evaluation against opponent pool...:", [p.name for p in players])
    for opponent in players:
        if opponent.is_mgr():
            pool = opponent.get_last_n_opponents(20)
            results = puffer_evaluate_against_pool(env, agent, pool, num_episodes, step, heatmap=heatmap)
            win_count = sum([results[op]['win'] for op in results])
            lose_count = sum([results[op]['lose'] for op in results])
            draw_count = sum([results[op]['draw'] for op in results])
            overall_stats[opponent.name] = {
                'win': win_count,
                'lose': lose_count,
                'draw': draw_count
            }
            opponent.record_last_n_scores([results[op]['win'] / num_episodes for op in results])
        else:
            win_count, lose_count, draw_count = puffer_evaluate(env, agent, opponent, num_episodes, step, heatmap=heatmap)
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
    env.close()


    NUM_ENVS = 8
    cfg.num_envs = NUM_ENVS
    vec_env = pfw.create_vec_env(backend='serial', num_envs=cfg.num_envs)

    opponent = pfw.VecBasicOpponent(cfg.num_envs, weak=True)

    EVAL_EPISODES = 16
    print(f"Evaluation over {EVAL_EPISODES} episodes: ")

    st_time = time.time()
    results = puffer_evaluate(
        vec_env,
        agent,
        opponent,
        num_episodes=EVAL_EPISODES
    )
    win_count, lose_count, draw_count = results
    print(f"Wins: {win_count}, Losses: {lose_count}, Draws  : {draw_count}.")


if __name__ == '__main__':
    import sys
    arguments = sys.argv[1:]
    main(arguments)
