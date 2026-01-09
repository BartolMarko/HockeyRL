import numpy as np
from agent import Agent
from evaluate import evaluate_env_bot
import hockey.hockey_env as h_env
import helper as h
import os
from pathlib import Path
import csv
from omegaconf import OmegaConf
from helper import Logger

def run_episode(agent, env, episode_index=0):
    """
    Runs one episode in the given environment with the specified agent.
    If 'opponent' is provided, uses self-play with that opponent.
    Otherwise, plays against the environment's built-in AI.

    Returns:
        (float) final score of the episode
    """
    obs = env.reset()[0]
    done = False
    truncated = False
    score = 0.0

    episode_metrics = {}
    losses = {}
    steps = 0

    while not done:
        agent_action = agent.choose_action(obs)

        obs_, reward, done, truncated, info = env.step(agent_action)

        done = done or truncated
        score += reward
        steps += 1


        agent.store(obs, agent_action, reward, obs_, done)
        loss = agent.learn(step=episode_index)
        if loss is not None:
            losses.update(loss)

        #env.render()
        obs = obs_

    episode_metrics['episode_score'] = score
    episode_metrics['episode_length'] = steps
    return episode_metrics, losses

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
        env_step = i + 1
        metrics, losses = run_episode(
            agent=agent,
            env=env,
            episode_index=i,
        )
        score_history.append(metrics['episode_score'])
        average_score = np.mean(score_history[-100:])

        len_history.append(metrics['episode_length'])
        average_length = np.mean(len_history[-100:])

        logger.add_scalar("Rollout/Avg Episode Score", average_score, i)
        logger.add_scalar("Rollout/Avg Episode Length", average_length, i)

        for key, value in losses.items():
            logger.add_scalar("Losses/" + key, value, i)

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
        print(f"Episode {i} completed. Recent Avg Score: {average_score:.2f}, Recent Avg Episode Length: {average_length:.2f}")

    logger.close()
    env.close()
    print("[Training completed]")

def set_dry_run_params(cfg):
    if cfg.get('dry_run', False):
        cfg.train_steps = 1000
        cfg.eval_freq = 50
        cfg.eval_episodes = 2
        cfg.save_model_freq = 900
        cfg.max_buffer_size = 100
        cfg.batch_size = 16
        cfg.hidden_dim = 16
        cfg.seed_steps = 10
        cfg.use_wandb = False
    return cfg

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        cfg = OmegaConf.load(f)
    cfg = set_dry_run_params(cfg)

    env = h_env.HockeyEnv_BasicOpponent(mode=0, weak_opponent=True)
    agent = Agent(
        lr_actor=cfg['lr_actor'],
        lr_critic=cfg['lr_critic'],
        input_dims=env.observation_space.shape,
        env=env,
        gamma=cfg['gamma'],
        n_actions=cfg['n_actions'],
        buffer_max_size=cfg['buffer_max_size'],
        hidden_size=cfg['hidden_size'],
        batch_size=cfg['batch_size'],
        reward_scale=cfg['reward_scale'],
        alpha=cfg['alpha'],
    )
    results_dir = Path(__file__).resolve().parent / "results" / cfg.exp_name
    logger = Logger(cfg, results_dir)
    logger.log_git_info()
    if cfg.log_gradients:
        logger.add_model(agent)
    train_agent(cfg, agent, env, logger)
