import os
import re
import gymnasium as gym
import hockey.hockey_env as h_env
import imageio
import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from opponents import OpponentInPool, OpponentPool
from pathlib import Path
from agent import Agent

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

class HeatmapTracker:
    def __init__(self, num_envs, x_range=(-4, 4), y_range=(-2.5, 2.5), bins=(48, 32)):
        """
        Tracks agent positioning and generates a heatmap.

        Args:
            num_envs: Number of parallel environments.
            x_range: Bounds of the hockey rink on the x-axis.
            y_range: Bounds of the hockey rink on the y-axis.
            bins: Resolution of the heatmap.
        """
        self.x_range = x_range
        self.y_range = y_range
        self.bins = bins
        self.num_envs = num_envs

        self.heatmap = np.zeros(bins)
        self.x_data = np.linspace(x_range[0], x_range[1], bins[0] + 1)
        self.y_data = np.linspace(y_range[0], y_range[1], bins[1] + 1)

        self.total_steps = 0

    def reset(self):
        self.heatmap = np.zeros(self.bins)
        self.total_steps = 0

    def increment_total_steps(self, n=None):
        self.total_steps += n if n is not None else self.num_envs

    def record(self, obs, idx, idy):
        assert self.num_envs == obs.shape[0], "Observation batch size does not match number of environments."
        ix = np.clip(np.digitize(obs[:, idx], self.x_data) - 1, 0, self.bins[0] - 1)
        iy = np.clip(np.digitize(obs[:, idy], self.y_data) - 1, 0, self.bins[1] - 1)
        np.add.at(self.heatmap, (ix, iy), 1)

    def save_heatmap(self, filename="agent_heatmap.png", title="Agent Position Density", cmap="seismic"):
        """
        Visualizes the accumulated histogram.
        """
        if np.sum(self.heatmap) == 0:
            print("[WARN] No data recorded yet for heatmap.")
            return

        plt.figure(figsize=(12, 7))
        extent = [self.x_range[0], self.x_range[1], self.y_range[0], self.y_range[1]]
        im = plt.imshow(
            self.heatmap.T,
            extent=extent,
            origin='lower',
            cmap=cmap,
            interpolation='gaussian'
        )

        cbar = plt.colorbar(im)
        cbar.set_label('# of Visits', rotation=270, labelpad=15)
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')

        self._draw_rink_elements()

        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        self.reset()

    def _draw_rink_elements(self):
        # Center red line
        plt.axvline(x=0, color='white', linestyle='-', linewidth=2, alpha=0.5)

        # Goal areas (approximate)
        goal_l = plt.Rectangle((-4, -0.7), 0.2, 1.4, color='white', alpha=0.3)
        goal_r = plt.Rectangle((3.8, -0.7), 0.2, 1.4, color='white', alpha=0.3)
        plt.gca().add_patch(goal_l)
        plt.gca().add_patch(goal_r)

    def compute_entropy(self):
        """
        Computes the Shannon entropy of the heatmap distribution.
        """
        prob_dist = self.heatmap / np.sum(self.heatmap)
        prob_dist = prob_dist[prob_dist > 0]
        shannon_entropy = -np.sum(prob_dist * np.log(prob_dist))
        return shannon_entropy

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
        self.agent_pos_heatmap = HeatmapTracker(cfg.num_envs)
        self.puck_pos_heatmap = HeatmapTracker(cfg.num_envs)
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

            print("[WNDB] wandb:")
            print("[WNDB] - Project:", self.cfg.get('wandb_project'))
            wandb_logger = wandb.init(project=self.cfg.get('wandb_project'),
                              name=self.cfg.get('exp_name'), config=dict(self.cfg),
                              monitor_gym=True, allow_val_change=True)
            return wandb
        return None

    def log_config(self):
        with open(self.project_dir / "config.yaml", 'w') as f:
            OmegaConf.save(self.cfg, f)

    def add_state(self, obs):
        """
        Saves the state for later visualization.
        """
        self.agent_pos_heatmap.record(obs, idx=0, idy=1)
        self.agent_pos_heatmap.record(obs, idx=6, idy=7)  # opponent
        self.puck_pos_heatmap.record(obs, idx=12, idy=13)
        self.agent_pos_heatmap.increment_total_steps()
        self.puck_pos_heatmap.increment_total_steps()

    def add_action(self, actions):
        """
        Saves historgram of actions.
        """
        for i in range(actions.shape[1]):
            self.add_historam(f"agent/action_dim_{i}", actions[:, i], bins=20)

    def log_heatmaps(self, step: int = 0):
        """
        Logs the saved states for an episode.
        """
        heatmap_title = "Player Positions"
        heatmap_folder = self.get_project_dir() / "heatmaps"
        heatmap_folder.mkdir(parents=True, exist_ok=True)
        heatmap_filename = heatmap_folder / f"player_positions_episode_{step}.png"
        self.agent_pos_heatmap.save_heatmap(filename=heatmap_filename,
                                           title=f"{heatmap_title} - Step {step}")
        if self.tb_logger is not None:
            self.tb_logger.add_image(f"heatmap/" + heatmap_title,
                                     plt.imread(heatmap_filename), dataformats='HWC')
        if self.wandb is not None:
            self.wandb.log({"heatmap/" + heatmap_title: wandb.Image(str(heatmap_filename))})

        heatmap_title = "Puck Positions"
        heatmap_filename = heatmap_folder / f"puck_positions_episode_{step}.png"
        self.puck_pos_heatmap.save_heatmap(filename=heatmap_filename,
                                          title=f"{heatmap_title} - Step {step}", cmap="Greens")
        if self.tb_logger is not None:
            self.tb_logger.add_image(f"heatmap/" + heatmap_title,
                                     plt.imread(heatmap_filename), dataformats='HWC')
        if self.wandb is not None:
            self.wandb.log({"heatmap/" + heatmap_title: wandb.Image(str(heatmap_filename))})

    def get_project_dir(self):
        return self.project_dir

    def add_model(self, agent):
        if self.wandb is not None:
            for model in agent.get_models():
                self.wandb.watch(model, log="all", log_freq=self.cfg.get('wandb_model_log_freq', 100))

    def add_scalar(self, key, value):
        if self.tb_logger is not None:
            self.tb_logger.add_scalar(key, value)
        if self.wandb is not None:
            self.wandb.log({key: value})
        self.data[key] = value

    def add_gif(self, key, gif_path, caption=""):
        if self.wandb is not None and gif_path is not None:
            self.wandb.log({key: wandb.Video(gif_path, caption=caption, format="gif")})

    def log_metrics(self, metrics: dict):
        if self.tb_logger is not None:
            for key, value in metrics.items():
                self.tb_logger.add_scalar(key, value)
        if self.wandb is not None:
            self.wandb.log(metrics)
        self.data.update(metrics)

    def add_historam(self, key, values, bins='auto'):
        if self.tb_logger is not None:
            self.tb_logger.add_histogram(key, values, bins=bins)
        if self.wandb is not None:
            self.wandb.log({key: wandb.Histogram(values)})
        self.data[key] = values

    def add_agent_artifacts(self, agent_state_folder: str, agent: Agent):
        if self.wandb is not None:
            artifact = self.wandb.Artifact(f'agent-{self.cfg.exp_name}', type='model')
            artifact.add_dir(agent_state_folder)
            self.wandb.log_artifact(artifact)

    def add_opponent_stats(self, opponent: OpponentInPool):
        stats = {
            f"Opponent/{opponent.name}/win_rate": opponent.get_win_rate(),
            f"Opponent/{opponent.name}/priority": opponent.priority
        }
        for key, value in stats.items():
            if self.tb_logger is not None:
                self.tb_logger.add_scalar(key, value)
            if self.wandb is not None:
                self.wandb.log({key: value})
            self.data[key] = value

    def add_opponent_pool_stats(self, opponent_pool: OpponentPool):
        weights = opponent_pool.sampler.get_weights()
        for idx, opponent in enumerate(opponent_pool.get_all_opponents()):
            opponent.priority = weights[idx] if idx < len(weights) else 0.0
            self.add_opponent_stats(opponent)

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
            }, allow_val_change=True)

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

def get_Nth_checkpoint(models_dir, nth_episode, prefix='episode_'):
    """Returns the folder path containing the N-th episode checkpoint."""
    nth_dir = os.path.join(models_dir, f"{prefix}{nth_episode}")
    return nth_dir

def set_env_params(cfg, env):
    if env == None:
        env = h_env.HockeyEnv()
    cfg.input_dims = env.observation_space.shape
    return cfg

def load_agent_from_config(experiment_name: str, env, inference_only=False) -> Agent:
    """Load an agent from a configuration file."""
    config_path = Path('results') / experiment_name / 'config.yaml'
    cfg = OmegaConf.load(config_path)
    cfg = set_env_params(cfg, env)
    cfg.resume = True
    agent = Agent(cfg, inference_only=inference_only)
    return agent

def load_agent_Nth_episode(experiment_name: str, n: int, env=None, resume=False, inference_only=False) -> Agent:
    """Load an agent from the N-th episode checkpoint."""
    config_path = Path('results') / experiment_name / 'config.yaml'
    cfg = OmegaConf.load(config_path)
    cfg = set_env_params(cfg, env)
    cfg.resume = resume
    agent = Agent(cfg, inference_only=inference_only)
    nth_checkpoint_dir = get_Nth_checkpoint(Path('results') / experiment_name / 'models', n)
    agent.load_models(nth_checkpoint_dir)
    return agent

def check_failure(logger: Logger, agent: Agent) -> str:
    """Checks if the previous run of the expirement failed."""
    project_dir = logger.get_project_dir()
    if not project_dir.exists():
        return ''
    # check if the config file exists
    config_path = project_dir / 'config.yaml'
    if not config_path.exists():
        return ''
    # check if the config file content matches the current agent config
    saved_cfg = OmegaConf.load(config_path)
    if saved_cfg != agent.cfg:
        print(f"[WARN] Config file mismatch with an older experiment. Assuming no failure.")
        return ''
    # check if failure folder exists in models
    failure_ckpt = project_dir / 'failure'
    if failure_ckpt.exists():
        print(f"[INFO] Detected failure from previous run. Using failure ckpt")
        return failure_ckpt
    latest_ckpt = get_latest_checkpoint(project_dir / 'models')
    if not os.path.exists(latest_ckpt):
        print(f"[WARN] No checkpoint found to load on failure.")
        return ''
    return latest_ckpt

def load_checkpoint_on_failure(agent: Agent, ckpt_dir: str):
    """Loads the latest checkpoint if a failure is detected."""
    if not os.path.exists(ckpt_dir):
        print(f"[WARN] No checkpoint found to load on failure.")
        return
    print(f"[INFO] Loading latest checkpoint from {ckpt_dir} due to failure.")
    agent.load_models(ckpt_dir, memory=True)
    # move the failure folder to avoid repeated loading
    failure_path = Path(ckpt_dir)
    new_path = failure_path.parent / f"recovered_{failure_path.name}"
    if new_path.exists():
        # remove existing recovered folder
        if new_path.is_dir():
            for item in new_path.iterdir():
                if item.is_dir():
                    os.rmdir(item)
                else:
                    os.remove(item)
            os.rmdir(new_path)
        else:
            os.remove(new_path)
    failure_path.rename(new_path)

def save_modules_on_failure(agent: Agent, save_dir: str):
    """Saves agent modules on failure."""
    save_path = Path(save_dir) / "failure"
    save_path.mkdir(parents=True, exist_ok=True)
    agent.save_models(save_path, memory=True)
    print(f"[INFO] Saved agent modules to {save_path} due to failure.")

if __name__ == '__main__':
    # test heatmap
    env = h_env.HockeyEnv()
    heatmap_logger = HeatmapTracker()
    obs = env.reset()
    for i in range(100000):
        agent_action = env.action_space.sample() if i % 10 == 0 else agent_action
        opponent_action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(np.hstack([agent_action, opponent_action]))
        heatmap_logger.record(obs, idx=0, idy=1)
        heatmap_logger.record(obs, idx=12, idy=13)
        heatmap_logger.increment_total_steps()
        if done:
            obs = env.reset()
    heatmap_logger.save_heatmap(filename='heatmap_test.png')
