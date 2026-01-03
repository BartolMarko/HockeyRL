import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

from hockey import hockey_env as h_env

from tdmpc import TDMPC
import imageio

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default.yaml"
MODEL_PATH = Path(__file__).resolve().parent / "results" / "baseline_sparse_reward" / "model_step_500013.pt"
GIF_SAVE_PATH = Path(__file__).resolve().parent / "results" / "baseline_sparse_reward" / "evaluation.gif"

def evaluate(env, agent, opponent, num_episodes, step, render=False):
	"""Evaluate a trained agent and optionally save a video."""
	win_count, lose_count, draw_count = 0, 0, 0
	frames = []
	for i in range(num_episodes):
		obs, _ = env.reset()
		obs_opponent = env.obs_agent_two()
		done = False
		while not done:
			if render:
				frames.append(env.render(mode='human'))
			action = agent.plan(obs, eval_mode=True, step=step).cpu().numpy()
			opponent_action = opponent.act(obs_opponent)
			obs, reward, done, _, info = env.step(np.hstack([action, opponent_action]))
			obs_opponent = env.obs_agent_two()
			reward -= info.get("reward_closeness_to_puck", 0)
		
		if reward < 0:
			lose_count += 1
		elif reward > 0:
			win_count += 1
		else:
			draw_count += 1

	# if render:
	# 	imageio.mimsave(GIF_SAVE_PATH, frames, fps=30)
	# 	print(f"Saved evaluation video to {GIF_SAVE_PATH}.")

	return win_count, lose_count, draw_count


def main():
    with open(CONFIG_PATH, 'r') as f:
        cfg = OmegaConf.load(f)

    env = h_env.HockeyEnv()
	
    cfg.obs_dim = env.observation_space.shape[0]
    cfg.obs_shape = env.observation_space.shape
    cfg.action_dim = env.action_space.shape[0] // 2

    agent = TDMPC(cfg)
    agent.load(MODEL_PATH)

    opponent = h_env.BasicOpponent(weak=False)

    # Evaluate agent
    win_count, lose_count, draw_count = evaluate(env, agent, opponent, 100, step=0, render=True)
    print(f"Evaluation over {cfg.eval_episodes} episodes: "
          f"Wins: {win_count}, Losses: {lose_count}, Draws: {draw_count}.\n")

if __name__ == '__main__':
	main()