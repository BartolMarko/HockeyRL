import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from torch.utils.tensorboard import SummaryWriter
import wandb
from omegaconf import OmegaConf
from opponents import OpponentInPool, OpponentPool

def get_tensor(x, device, dtype=torch.float32):
    """Converts input to a torch tensor on the specified device."""
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    if isinstance(x, torch.Tensor):
        x = x.to(device=device, dtype=dtype)
    return x

def linear_schedule(schdl, step):
    """
    Outputs values following a linear decay schedule.
    Adapted from https://github.com/facebookresearch/drqv2
    """
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
    raise NotImplementedError(schdl)

class Logger:
    """
    Simple logger class to store training statistics.
    Logs a dictionary for tensorboard logging and if cfg provides wandb, use that as well.
    """
    def __init__(self, cfg, project_dir):
        self.cfg = cfg
        self.project_dir = project_dir
        project_dir.mkdir(parents=True, exist_ok=True)
        self.tb_logger = SummaryWriter(project_dir / 'logs')
        self.wandb = self._init_wandb()
        assert self.tb_logger is not None or self.wandb is not None, "No logging method specified."
        self.data = {}
        self.log_config()

    def _init_wandb(self):
        if self.cfg.get('use_wandb', False):
            if os.getenv('WANDB_API_KEY') is None:
                raise ValueError("WANDB_API_KEY environment variable not set.")
            if self.cfg.resume:
                if self.cfg.get('wandb_run_id') is not None:
                    os.environ['WANDB_RUN_ID'] = self.cfg.get('wandb_run_id')
                os.environ['WANDB_RESUME'] = self.cfg.get('wandb_resume', 'allow')
                if os.getenv('WANDB_RESUME') not in ['must', 'auto', 'allow']:
                    raise ValueError("To resume a wandb run, set wandb_resume='must' or 'auto'.")
                if not os.getenv('WANDB_RUN_ID'):
                    raise ValueError("To resume a wandb run, set wandb_run_id in the config.")

            print("wandb:")
            print(" - Project:", self.cfg.get('wandb_project'))
            wandb_logger = wandb.init(project=self.cfg.get('wandb_project'),
                              name=self.cfg.get('exp_name'), config=dict(self.cfg),
                              monitor_gym=True)
            if self.cfg.resume:
                wandb.config.update(dict(self.cfg), allow_val_change=True)
        return None


    def log_config(self):
        with open(self.project_dir / "config.yaml", 'w') as f:
            OmegaConf.save(self.cfg, f)

    def get_project_dir(self):
        return self.project_dir

    def add_model(self, agent):
        if self.wandb is not None:
            for model in agent.get_models():
                self.wandb.watch(model, log="all", log_freq=self.cfg.get('wandb_model_log_freq', 100))

    def add_scalar(self, key, value, step):
        if self.tb_logger is not None:
            self.tb_logger.add_scalar(key, value, step)
        if self.wandb is not None:
            self.wandb.log({key: value}, step=step)
        self.data[key] = (step, value)

    def add_gif(self, key, gif_path, step, caption=""):
        if self.wandb is not None and gif_path is not None:
            self.wandb.log({key: wandb.Video(gif_path, caption=caption, format="gif")}, step=step)

    def add_historam(self, key, values, step, bins='auto'):
        if self.tb_logger is not None:
            self.tb_logger.add_histogram(key, values, step, bins=bins)
        if self.wandb is not None:
            self.wandb.log({key: wandb.Histogram(values)}, step=step)
        self.data[key] = (step, values)

    def add_opponent_stats(self, opponent: OpponentInPool, step: int):
        stats = {
            f"Opponent/{opponent.name}/win_rate": opponent.win_count / max(1, opponent.get_games_played()),
            f"Opponent/{opponent.name}/priority": opponent.priority
        }
        for key, value in stats.items():
            if self.tb_logger is not None:
                self.tb_logger.add_scalar(key, value, step)
            if self.wandb is not None:
                self.wandb.log({key: value}, step=step)
            self.data[key] = (step, value)

    def add_opponent_pool_stats(self, opponent_pool: OpponentPool, step: int):
        for opponent in opponent_pool.get_all_opponents():
            self.add_opponent_stats(opponent, step)

    def get_logs(self):
        return self.data

    def clear(self):
        self.data = {}

    def log_git_info(self, filename='git-commit-hash.txt'):
        commit_info = read_commit_info(filename)
        # Log git commit info as text information
        if self.tb_logger is not None:
            self.tb_logger.add_text('git-commit-hash', commit_info['git-commit-hash'], 0)
            self.tb_logger.add_text('git-commit-log', commit_info['git-commit-log'], 0)
        if self.wandb is not None:
            self.wandb.config.update({
                'git-commit-hash': commit_info['git-commit-hash'],
                'git-commit-log': commit_info['git-commit-log']
            })

    def close(self):
        if self.tb_logger is not None:
            self.tb_logger.close()
        if self.wandb is not None:
            self.wandb.finish()

def read_commit_info(filename='git-commit-hash.txt'):
    """Reads git commit info from a file."""
    commit_info = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                key, value = line.strip().split(': ', 1)
                commit_info[key] = value
    except FileNotFoundError:
        commit_info = {
            'git-commit-hash': 'N/A',
            'git-commit-log': 'N/A'
        }
    return commit_info

def get_resume_episode_number(models_dir, prefix='episode_'):
    """Returns the latest episode number for resuming training."""
    episode_dirs = [d for d in os.listdir(models_dir) if d.startswith(prefix)]
    if not episode_dirs:
        return 0
    latest_episode = max(int(d.split('_')[1]) for d in episode_dirs)
    return latest_episode

def get_latest_checkpoint(models_dir, prefix='episode_'):
    """Returns the folder path containing the latest checkpoint."""
    latest_episode = get_resume_episode_number(models_dir, prefix)
    latest_dir = os.path.join(models_dir, f"{prefix}{latest_episode}")
    return latest_dir

def set_env_params(cfg, env):
    cfg.input_dims = env.observation_space.shape
    return cfg
