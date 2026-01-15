import numpy as np
from agent import Agent
from evaluate import evaluate_env_bot
import hockey.hockey_env as h_env
import helper as h
import os
from pathlib import Path
import csv
from omegaconf import OmegaConf
from helper import Logger, set_env_params
from rewards import RewardShaper
AVG_WINDOW_SIZE = 25

def run_episode(cfg, agent, env, episode_index=None):
    reward_shaper = RewardShaper(cfg)
    obs = env.reset()[0]
    done = False
    truncated = False
    score = 0.0

    episode_metrics = {}
    steps = 0

    while not done:
        agent_action = agent.choose_action(obs, episode_index)

        obs_, reward, done, truncated, info = env.step(agent_action)
        reward = reward_shaper.transform(reward, info, done or truncated)

        done = done or truncated
        score += reward
        steps += 1

        agent.store(obs, agent_action, reward, obs_, done)
        obs = obs_

    episode_metrics['episode_score'] = score
    episode_metrics['episode_length'] = steps
    return episode_metrics

def train_agent(cfg, agent, env, logger):
    n_games = cfg.n_games

    gif_save_path = None
    if cfg.get('save_video', False):
        gif_save_dir = logger.get_project_dir() / 'videos'
        gif_save_dir.mkdir(parents=True, exist_ok=True)

    last_save_step = 0
    last_eval_step = 0
    env_step = 0

    score_history = []
    len_history = []

    for i in range(n_games):
        metrics = run_episode(
            cfg=cfg,
            agent=agent,
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
                metrics = agent.learn(step=i)
                if metrics is not None:
                    __import__('pdb').set_trace()
                    for key, value in metrics.items():
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
            eval_win, eval_lose, eval_draw = evaluate_env_bot(
                env=env,
                agent=agent,
                num_episodes=cfg.eval_episodes,
                step=i,
                render=False,
                save=gif_save_path
            )
            logger.add_scalar("Eval/Win Rate", eval_win / cfg.eval_episodes, i)
            logger.add_scalar("Eval/Lose Rate", eval_lose / cfg.eval_episodes, i)
            logger.add_scalar("Eval/Draw Rate", eval_draw / cfg.eval_episodes, i)
            logger.add_gif("Eval/Episode", gif_save_path, i)
            print(f"Evaluation at Episode {i}: Win: {eval_win}, Lose: {eval_lose}, Draw: {eval_draw}")
        env_step = i + 1
        print(f"Episode {i} completed. Recent Avg Score: {average_score:.2f}, Recent Avg Episode Length: {average_length:.2f}")

    logger.close()
    env.close()
    print("[Training completed]")

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
        cfg.exp_name = f"dry_run_{cfg.exp_name}"
    return cfg

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        cfg = OmegaConf.load(f)
    cfg = set_dry_run_params(cfg)
    env = h_env.HockeyEnv_BasicOpponent(mode=0, weak_opponent=cfg.weak_opponent)
    cfg = set_env_params(cfg, env)
    agent = Agent(cfg)
    results_dir = Path(__file__).resolve().parent / "results" / cfg.exp_name
    logger = Logger(cfg, results_dir)
    logger.log_git_info()
    if cfg.log_gradients:
        logger.add_model(agent)
    train_agent(cfg, agent, env, logger)
