import torch
import numpy as np
import time
import random
from pathlib import Path
from hockey import hockey_env as h_env
from omegaconf import OmegaConf
torch.backends.cudnn.benchmark = True
from tdmpc import TDMPC
from helper import Episode, ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default.yaml"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

SUMMARY_WRITER = SummaryWriter(RESULTS_DIR / "logs")

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def add_env_variables_to_config(env, cfg):
	"""Add environment-specific variables to the config."""
	cfg.obs_dim = env.observation_space.shape[0]
	cfg.obs_shape = env.observation_space.shape

	cfg.action_dim = env.action_space.shape[0] // 2
	return cfg


def evaluate(env, agent, opponent, num_episodes, step, env_step):
	"""Evaluate a trained agent and optionally save a video."""
	win_count, lose_count, draw_count = 0, 0, 0
	for i in range(num_episodes):
		obs, _ = env.reset()
		obs_opponent = env.obs_agent_two()
		done = False
		episode_reward = 0
		while not done:
			action = agent.plan(obs, eval_mode=True).cpu().numpy()
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

	return win_count, lose_count, draw_count


def train(cfg):
	"""Training script for TD-MPC. Requires a CUDA-enabled device."""
	assert torch.cuda.is_available()
	set_seed(cfg.seed)
	env = h_env.HockeyEnv()
	cfg = add_env_variables_to_config(env, cfg)
	agent, buffer = TDMPC(cfg), ReplayBuffer(cfg)
	opponent = h_env.BasicOpponent(weak=False)
	
	# Run training
	episode_idx, start_time = 0, time.time()
	last_update_step, last_eval_step = 0, 0
	step = 0
	while step < cfg.train_steps:
		# Collect trajectory
		obs, _ = env.reset()
		obs_opponent = env.obs_agent_two()
		episode = Episode(cfg, obs)
		while not episode.done:
			step += 1
			action = agent.plan(obs, step=step, t0=episode.first).cpu().numpy()
			opponent_action = opponent.act(obs_opponent)
			obs, reward, done, _, _ = env.step(np.hstack([action, opponent_action]))
			obs_opponent = env.obs_agent_two()
			episode += (obs, action, opponent_action, reward, done)
		buffer += episode

		# Update model
		train_metrics = {}
		if step >= cfg.seed_steps:
			num_updates = step - last_update_step
			last_update_step = step
			for i in range(num_updates):
				train_metrics.update(agent.update(buffer, step+i))

		# Log training metrics
		episode_idx += 1
		env_step = int(step*cfg.action_repeat)
		for k, v in train_metrics.items():
			SUMMARY_WRITER.add_scalar(f'Losses/{k}', v, env_step)
		
		print(f"Step {step}. Episode {episode_idx} finished in {time.time() - start_time:.2f}s.")

		# Evaluate agent periodically
		if env_step - last_eval_step >= cfg.eval_freq:
			win_count, lose_count, draw_count = evaluate(env, agent, opponent, cfg.eval_episodes, step, env_step)
			SUMMARY_WRITER.add_scalar('Eval/Win Rate', win_count / cfg.eval_episodes, env_step)
			SUMMARY_WRITER.add_scalar('Eval/Lose Rate', lose_count / cfg.eval_episodes, env_step)
			SUMMARY_WRITER.add_scalar('Eval/Draw Rate', draw_count / cfg.eval_episodes, env_step)
			print(f"Evaluation at step {step} ({env_step} env steps): "
				  f"Wins: {win_count}, Losses: {lose_count}, Draws: {draw_count}.\n")

			last_eval_step = env_step

	print('Training completed successfully')


if __name__ == '__main__':
	with open(CONFIG_PATH, 'r') as f:
		cfg = OmegaConf.load(f)
		train(cfg)