import numpy as np
from agent import Agent
from evaluate import evaluate, evaluate_against_pool
import hockey.hockey_env as h_env
import helper as h
import opponents as opp
import os
from pathlib import Path
import csv
from omegaconf import OmegaConf
from helper import Logger, set_env_params, get_resume_episode_number
from rewards import RewardShaper
AVG_WINDOW_SIZE = 25

def run_episode(cfg, agent, opponent, env, episode_index=None):
    reward_shaper = RewardShaper(cfg)
    obs = env.reset()[0]
    obs_opponent = env.obs_agent_two()
    done = False
    truncated = False
    score = 0.0

    episode_metrics = {}
    steps = 0

    while not done:
        opponent_action = opponent.get_step(obs_opponent)
        agent_action = agent.choose_action(obs, episode_index)

        obs_, reward, done, truncated, info = env.step(np.hstack([agent_action, opponent_action]))
        reward = reward_shaper.transform(reward, info, done or truncated)

        done = done or truncated
        score += reward
        steps += 1

        agent.store(obs, agent_action, reward, obs_, done)
        obs = obs_
        obs_opponent = env.obs_agent_two()

    episode_metrics['episode_score'] = score
    episode_metrics['episode_length'] = steps
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

    score_history = []
    len_history = []

    for i in range(n_games):
        opponent = opponent_pool.sample_opponent()
        metrics = run_episode(
            cfg=cfg,
            agent=agent,
            opponent=opponent,
            env=env,
            episode_index=i,
        )
        score_history.append(metrics['episode_score'])
        average_score = np.mean(score_history[-AVG_WINDOW_SIZE:])

        len_history.append(metrics['episode_length'])
        average_length = np.mean(len_history[-AVG_WINDOW_SIZE:])

        logger.add_scalar("Rollout/Episode Score", metrics['episode_score'], i)
        logger.add_scalar("Rollout/Episode Length", metrics['episode_length'], i)

        if i >= cfg.warmup_games:
            for _ in range(cfg.learn_steps_per_episode):
                l_metrics = agent.learn(step=i)
                if l_metrics is not None:
                    for key, value in l_metrics.items():
                        if 'hist:' in key:
                            logger.add_historam(key.replace('hist:', ''), value, i)
                        else:
                            logger.add_scalar(key, value, i)

        # save models if we have a new "best" average score
        if cfg.save_model and \
                ((env_step - last_save_step >= cfg.save_model_freq) or \
                (i == n_games - 1)):
            project_dir = logger.get_project_dir()
            models_dir = os.path.join(project_dir, 'models', f'episode_{i}')
            os.makedirs(models_dir, exist_ok=True)
            agent.save_models(models_dir)
            last_save_step = env_step
            print(f"Saved models at Episode {i} to {models_dir}.")

        if env_step - last_eval_step >= cfg.eval_freq:
            last_eval_step = env_step
            if gif_save_dir is not None:
                gif_save_path = gif_save_dir / f'eval_episode_{i}.gif'
            else:
                gif_save_path = None
            eval_win, eval_lose, eval_draw = evaluate(
                env=env,
                agent=agent,
                opponent=opponent,
                num_episodes=cfg.eval_episodes,
                step=i,
                render=False,
                save=gif_save_path
            )
            logger.add_gif("Eval/Episode", gif_save_path, i, caption=f"{opponent.name}")
            final_eval = evaluate_against_pool(env, agent, opponent_pool, num_episodes=cfg.eval_episodes)
            logger.add_opponent_pool_stats(opponent_pool, i)
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
            logger.add_scalar("Eval/Win Rate", win_rate, i)
            logger.add_scalar("Eval/Lose Rate", lose_rate, i)
            logger.add_scalar("Eval/Draw Rate", draw_rate, i)
            print(f"Evaluation at Episode {i}: Win: {win_rate}, Lose: {lose_rate}, Draw: {draw_rate}")
        env_step = i + 1
        print(f"Episode {i} completed. Recent Avg Score: {average_score:.2f}, Recent Avg Episode Length: {average_length:.2f}")

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
        cfg.warmup_games = 10
        cfg.eval_freq = 5
        cfg.eval_episodes = 5
        cfg.save_model_freq = 900
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
    train_agent(cfg, agent, env, logger, start_episode=start_episode)
