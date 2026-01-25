import numpy as np
import time
from agent import Agent
from evaluate import evaluate, evaluate_against_pool
import hockey.hockey_env as h_env
import puffer_wrapper as pfw
import evaluate_puffer as epfw
import helper as h
import opponents as opp
import os
from pathlib import Path
import csv
from omegaconf import OmegaConf
from helper import Logger, set_env_params, get_resume_episode_number
from rewards import RewardShaper
AVG_WINDOW_SIZE = 25

def run_episode(cfg, agent, opponent, vec_env, logger=None, episode_index=None):
    start_time = time.time()
    reward_shaper = RewardShaper(cfg)

    episodes_per_env = 1
    num_episodes = vec_env.num_envs * episodes_per_env

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
        if logger is not None: logger.add_state(obs_batch)

        agent_actions = agent.plan_batch(obs_batch, eval_mode=True)
        opponent_actions = opponent.get_step(obs_opponent_batch)
        combined_actions = np.hstack([agent_actions, opponent_actions])

        # IMP: Ensure float32 and clip to valid range
        combined_actions = np.clip(combined_actions, -1, 1).astype(np.float32)

        obs_batch, reward_batch, done_batch, _, info_batch = vec_env.step(combined_actions)
        rewards_batch = reward_shaper.transform_batch(reward_batch, info_batch, done_batch, obs_batch)
        obs_opponent_batch = vec_env.obs_agent_two()

        active_envs = (episodes_done < episodes_per_env)
        done_batch = np.logical_and(done_batch, active_envs)
        win_counts += (done_batch & (reward_batch > 0)).astype(int)
        lose_counts += (done_batch & (reward_batch < 0)).astype(int)
        draw_counts += (done_batch & (reward_batch == 0)).astype(int)

        episodes_done += done_batch.astype(int)
        done_count += np.sum(done_batch)

        episode_scores += rewards_batch.sum()
        episode_lengths += np.sum(active_envs)

        agent.store(obs_batch, agent_actions, rewards_batch, obs_batch, done_batch)

    agent.end_episode()
    episode_metrics = {}
    episode_metrics['episode_score'] = episode_scores / num_episodes
    episode_metrics['episode_length'] = episode_lengths / num_episodes
    end_time = time.time()
    episode_metrics['episode_time'] = ( end_time - start_time ) / num_episodes
    return episode_metrics

def train_agent(cfg, agent, env, logger, start_episode=0):
    n_games = cfg.n_games

    gif_save_path = None
    if cfg.get('save_video', False):
        gif_save_dir = logger.get_project_dir() / 'videos'
        gif_save_dir.mkdir(parents=True, exist_ok=True)

    last_save_step = 0
    last_eval_step = 0
    env_step = 0

    opponent_pool = opp.get_opponent_pool(cfg)

    agent.show_info()
    opponent_pool.show_info()

    score_history = []
    len_history = []

    for i in range(n_games):
        opponent = opponent_pool.sample_opponent()
        metrics = run_episode(
            cfg=cfg,
            agent=agent,
            opponent=opponent,
            vec_env=env,
            logger=logger,
            episode_index=i + start_episode,
        )
        score_history.append(metrics['episode_score'])
        average_score = np.mean(score_history[-AVG_WINDOW_SIZE:])

        len_history.append(metrics['episode_length'])
        average_length = np.mean(len_history[-AVG_WINDOW_SIZE:])

        print(f"Episode {i} vs {opponent.get_agent_name()}. Recent Avg Score: {average_score:.2f}, Recent Avg Episode Length: {average_length:.2f}")
        logger.add_scalar("Rollout/Episode Score", metrics['episode_score'])
        logger.add_scalar("Rollout/Episode Length", metrics['episode_length'])
        logger.add_scalar("Rollout/Episode Time", metrics['episode_time'])

        if i == cfg.warmup_games:
            print(f"[WARM] Warmup phase completed (step: {i}). Starting learning...")
            logger.log_state(i)

        if i >= cfg.warmup_games:
            for _ in range(int(cfg.learn_steps_per_episode * metrics['episode_length'])):
                l_metrics = agent.learn(step=i)
                if l_metrics is not None:
                     scalars = {}
                     histograms = {}
                     for key, value in l_metrics.items():
                         if 'hist:' in key:
                             histograms[key.replace('hist:', '')] = value
                         else:
                             scalars[key] = value

                     if scalars:
                        logger.log_metrics(scalars)
                     for key, value in histograms.items():
                        logger.add_historam(key, value)

        # save models
        if cfg.save_model and \
                ((env_step - last_save_step >= cfg.save_model_freq) or \
                (i == n_games - 1)) or \
                opponent_pool.self_play_mgr_needs_save(agent, env_step):
            project_dir = logger.get_project_dir()
            models_dir = os.path.join(project_dir, 'models', f'episode_{i}')
            os.makedirs(models_dir, exist_ok=True)
            agent.save_models(models_dir)
            last_save_step = env_step
            print(f"[SAVE] Saved models at Episode {i} to {models_dir}.")

        if env_step - last_eval_step >= cfg.eval_freq:
            last_eval_step = env_step
            if cfg.num_envs > 1:
                eval_fn = epfw.puffer_evaluate
            else:
                eval_fn = evaluate
            eval_win, eval_lose, eval_draw = eval_fn(
                env=env,
                agent=agent,
                opponent=opponent,
                num_episodes=cfg.eval_episodes,
                step=i,
                render=False,
                save=None
            )
            # logger.add_gif("Eval/Episode", gif_save_path, caption=f"{opponent.name}")
            if cfg.num_envs > 1:
                eval_fn = epfw.puffer_evaluate_against_pool
            else:
                eval_fn = evaluate_against_pool
            final_eval = eval_fn(env, agent, opponent_pool, num_episodes=cfg.eval_episodes)
            logger.add_opponent_pool_stats(opponent_pool)
            win_rate, lose_rate, draw_rate = 0.0, 0.0, 0.0
            total_episodes = 0
            for stats in final_eval.values():
                win_rate += stats['win']
                lose_rate += stats['lose']
                draw_rate += stats['draw']
            total_episodes = len(final_eval) * cfg.eval_episodes
            win_rate /= total_episodes
            lose_rate /= total_episodes
            draw_rate /= total_episodes
            logger.add_scalar("Eval/Win Rate", win_rate)
            logger.add_scalar("Eval/Lose Rate", lose_rate)
            logger.add_scalar("Eval/Draw Rate", draw_rate)
            logger.log_state(i)
            opponent_pool.show_scoreboard()
            opponent_pool.end_evaluation()
            print(f"[EVAL] Evaluation at Episode {i}: Win: {win_rate:.2f}, Lose: {lose_rate:.2f}, Draw: {draw_rate:.2f}")

        # every self play mgr handles the update by itself
        opponent_pool.update_pool(agent, episode_index=i, logger=logger)
        env_step = i + 1

    logger.close()
    env.close()
    print("[Training completed] [episodes: {} + {} = {}]".format(start_episode, env_step, env_step + start_episode))
    final_eval = evaluate_against_pool(env, agent, opponent_pool, num_episodes=cfg.eval_episodes)
    print("Opponents Stats:")
    opponent_pool.show_scoreboard()
    print("Agent Stats:")
    win_rate, lose_rate, draw_rate = 0.0, 0.0, 0.0
    total_episodes = 0
    for stats in final_eval.values():
        win_rate += stats['win']
        lose_rate += stats['lose']
        draw_rate += stats['draw']
    total_episodes = len(final_eval) * cfg.eval_episodes
    win_rate /= total_episodes
    lose_rate /= total_episodes
    draw_rate /= total_episodes
    print(f"Win Rate: {win_rate:.2f}, Lose Rate: {lose_rate:.2f}, Draw Rate: {draw_rate:.2f}")
    print(f"{win_rate:.2f}, {lose_rate:.2f}, {draw_rate:.2f}")


def set_dry_run_params(cfg):
    if cfg.get('dry_run', False):
        cfg.n_games = 500
        cfg.warmup_games = 4
        cfg.eval_freq = 2
        cfg.eval_episodes = 2
        cfg.save_model_freq = 1
        cfg.max_buffer_size = 128
        cfg.batch_size = 16
        cfg.hidden_dim = 16
        cfg.use_wandb = False
        if not cfg.resume:
            cfg.exp_name = f"dry_run_{cfg.exp_name}"
    return cfg

if __name__ == '__main__':
    start_episode = 0
    with open('config.yaml', 'r') as f:
        cfg = OmegaConf.load(f)
    cfg = set_dry_run_params(cfg)
    env = h_env.HockeyEnv()
    cfg = set_env_params(cfg, env)
    agent = Agent(cfg)
    results_dir = Path(__file__).resolve().parent / "results" / cfg.exp_name
    logger = Logger(cfg, results_dir)
    logger.log_git_info()
    if cfg.log_gradients:
        logger.add_model(agent)
    if cfg.resume:
        start_episode = get_resume_episode_number(results_dir / 'models')
        print(f"Resume training from Episode {start_episode}.")
    if hasattr(cfg, 'num_envs') and cfg.num_envs > 1:
        vec_env = pfw.create_vec_env(backend='multiprocessing', num_envs=cfg.num_envs)
        env = pfw.HockeyVecEnv(vec_env)
    train_agent(cfg, agent, env, logger, start_episode=start_episode)
