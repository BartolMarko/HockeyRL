import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

from hockey import hockey_env as h_env

from agent import Agent
import imageio

def evaluate(env, agent, opponent, num_episodes, step, render=False, save=None):
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
            action = agent.plan(obs, eval_mode=True, step=step)
            if not isinstance(action, np.ndarray):
                action = action.cpu().numpy()
            opponent_action = opponent.act(obs_opponent)
            obs, reward, done, _, info = env.step(np.hstack([action, opponent_action]))
            obs_opponent = env.obs_agent_two()

        if reward < 0:
            lose_count += 1
        elif reward > 0:
            win_count += 1
        else:
            draw_count += 1

    if save is not None:
        imageio.mimsave(save, frames, fps=30)
        print(f"Saved evaluation video to {GIF_SAVE_PATH}.")

    return win_count, lose_count, draw_count

def evaluate_env_bot(env, agent, num_episodes, step, render=False, save=None):
    """Evaluate a trained agent against the built-in environment bot and optionally save a video."""
    win_count, lose_count, draw_count = 0, 0, 0
    frames = []
    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            if render or save is not None:
                mode = 'human' if render else 'rgb_array'
                frames.append(env.render(mode=mode))
            action = agent.plan(obs, eval_mode=True, step=step)
            if not isinstance(action, np.ndarray):
                action = action.cpu().numpy()
            obs, reward, done, _, info = env.step(action)

        if reward < 0:
            lose_count += 1
        elif reward > 0:
            win_count += 1
        else:
            draw_count += 1

    if save is not None:
        imageio.mimsave(save, frames, fps=30)
        print(f"Saved evaluation video to {save}.")

    return win_count, lose_count, draw_count


# def main():
#     with open(CONFIG_PATH, 'r') as f:
#         cfg = OmegaConf.load(f)
#
#     env = h_env.HockeyEnv()
#     cfg.obs_dim = env.observation_space.shape[0]
#     cfg.obs_shape = env.observation_space.shape
#     cfg.state_dim = env.observation_space.shape[0]
#     cfg.opponent_action_dim = env.action_space.shape[0] // 2
#
#     if cfg.device == 'cuda' and not torch.cuda.is_available():
#         cfg.device = 'cpu'
#     agent = Agent(cfg)
#     agent.load(MODEL_PATH)
#
#     opponent = h_env.BasicOpponent(weak=False)
#
#     # Evaluate agent
#     win_count, lose_count, draw_count = evaluate(env, agent, opponent, 100, step=0, render=True, save=GIF_SAVE_PATH)
#     print(f"Evaluation over {cfg.eval_episodes} episodes: "
#           f"Wins: {win_count}, Losses: {lose_count}, Draws: {draw_count}.\n")
#
# if __name__ == '__main__':
#     main()
