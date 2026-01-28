import numpy as np
from omegaconf import OmegaConf
import time
import helper
from hockey import hockey_env as h_env
import puffer_wrapper as pfw

def evaluate(cfg, agent, opponent, vec_env, logger=None, episode_index=None, episodes_per_env=None):
    start_time = time.time()
    agent.eval()

    if episodes_per_env is None:
        episodes_per_env = 1
    num_episodes = vec_env.num_envs * episodes_per_env
    assert num_episodes > 0, "Number of episodes must be positive. episodes_per_env = {}".format(episodes_per_env)

    win_counts = np.zeros(vec_env.num_envs, dtype=int)
    lose_counts = np.zeros(vec_env.num_envs, dtype=int)
    draw_counts = np.zeros(vec_env.num_envs, dtype=int)
    episodes_done = np.zeros(vec_env.num_envs, dtype=int)
    obs_batch, _ = vec_env.reset()
    obs_opponent_batch = vec_env.obs_agent_two()
    done_batch = np.zeros(vec_env.num_envs, dtype=bool)
    active_envs = np.ones(vec_env.num_envs, dtype=bool)
    done_count = 0

    episode_scores = 0.0
    episode_lengths = 0

    while done_count < num_episodes:
        agent_actions = agent.plan_batch(obs_batch, eval_mode=True, step=episode_index)
        if logger is not None:
            logger.add_state(obs_batch)
            logger.add_action(agent_actions)

        opponent_actions = opponent.plan_batch(obs_opponent_batch)
        combined_actions = np.hstack([agent_actions, opponent_actions])

        # IMP: Ensure float32 and clip to valid range
        combined_actions = np.clip(combined_actions, -1, 1).astype(np.float32)

        obs_batch, rewards_batch, done_batch, truncated_batch, info_batch = vec_env.step(combined_actions)

        active_envs = (episodes_done < episodes_per_env)
        # We should count result if it is terminated OR truncated
        real_done_batch = np.logical_or(done_batch, truncated_batch)
        finished_batch = np.logical_and(real_done_batch, active_envs)
        if 'final_info' in info_batch:
            for idx, infos in enumerate(info_batch['final_info']):
                if infos is None: continue
                winner = infos.get('winner')
                if winner == 1:
                    win_counts[idx] += 1
                elif winner == -1:
                    lose_counts[idx] += 1
                elif winner == 0:
                    draw_counts[idx] += 1
        else:
            # unlikely, but lets keep this
            win_counts += (finished_batch & (rewards_batch > 0)).astype(int)
            lose_counts += (finished_batch & (rewards_batch < 0)).astype(int)
            draw_counts += (finished_batch & (rewards_batch == 0)).astype(int)

        episodes_done += finished_batch.astype(int)
        done_count += np.sum(finished_batch)

        if isinstance(vec_env, pfw.HockeyVecEnv): # Ensure we are using the wrapper
             obs_opponent_batch = vec_env.obs_agent_two()
        else:
             obs_opponent_batch = info_batch.get('obs_agent_two') # Fallback

        episode_scores += rewards_batch.sum()
        episode_lengths += np.sum(active_envs)

    if hasattr(opponent, 'record_play_scores'):
        total_wins = int(np.sum(win_counts))
        total_losses = int(np.sum(lose_counts))
        total_draws = int(np.sum(draw_counts))
        opponent.record_play_scores(total_losses, total_wins, total_draws)
    total_episodes = np.sum(win_counts) + np.sum(lose_counts) + np.sum(draw_counts)
    episode_metrics = {}
    episode_metrics['episode_score'] = episode_scores / total_episodes
    episode_metrics['episode_length'] = episode_lengths / total_episodes
    episode_metrics['win'] = np.sum(win_counts)
    episode_metrics['lose'] = np.sum(lose_counts)
    episode_metrics['draw'] = np.sum(draw_counts)
    episode_metrics['total_episodes'] = total_episodes
    end_time = time.time()
    episode_metrics['episode_time'] = ( end_time - start_time ) / total_episodes
    return episode_metrics

def puffer_evaluate_against_pool(env, agent, opponent_pool, num_episodes: int = 100, step: int | None = None, logger=None):
    overall_stats = {}
    players = opponent_pool.get_playable_opponents()
    print("[EVAL] Starting evaluation against opponent pool...:", [p.name for p in players])
    for opponent in players:
        if opponent.is_mgr():
            pool = opponent.get_last_n_opponents(20)
            results = puffer_evaluate_against_pool(
                    env, agent, pool,
                    num_episodes=num_episodes,
                    logger=logger,
                    step=step)
            win_count = sum([results[op]['win'] for op in results])
            lose_count = sum([results[op]['lose'] for op in results])
            draw_count = sum([results[op]['draw'] for op in results])
            overall_stats[opponent.name] = {
                'win': win_count,
                'lose': lose_count,
                'draw': draw_count
            }
        else:
            metrics = evaluate(agent.cfg, agent, opponent, env, episode_index=step, logger=logger)
            overall_stats[opponent.name] = {
                'win': metrics['win'],
                'lose': metrics['lose'],
                'draw': metrics['draw']
            }
            if opponent.pool:
                opponent.pool.record_play_scores(metrics['lose'], metrics['win'], metrics['draw'], episode_index=opponent.ep)
        print(f"[EVAL] vs {opponent.name}: Wins: {overall_stats[opponent.name]['win']}, Losses: {overall_stats[opponent.name]['lose']}, Draws: {overall_stats[opponent.name]['draw']}.")
    return overall_stats


def main(args):
    # if len(args) < 1:
    #     with open('config.yaml', 'r') as f:
    #         cfg = OmegaConf.load(f)
    #     experiment_name = cfg.exp_name
    # else:
    #     experiment_name = args[0]
    experiment_name = 'v4-per-6_4-bot-pool'
    env = h_env.HockeyEnv()
    agent = helper.load_agent_from_config(experiment_name, env)
    env.close()


    NUM_ENVS = 8
    vec_env = pfw.create_vec_env(backend='serial', num_envs=NUM_ENVS)

    opponent = pfw.VecBasicOpponent(NUM_ENVS, weak=True)

    EP_PER_ENV = 5
    EVAL_EPISODES = NUM_ENVS * EP_PER_ENV
    print(f"Evaluation over {EVAL_EPISODES} episodes: ")

    st_time = time.time()
    metrics = evaluate(agent.cfg, agent, opponent, vec_env, episodes_per_env=EP_PER_ENV)
    print(metrics)


if __name__ == '__main__':
    import sys
    arguments = sys.argv[1:]
    main(arguments)
