import torch
import wandb
import numpy as np

from omegaconf import OmegaConf
from pathlib import Path

from src.named_agent import NamedAgent
from src.ensembles.ppo import PPO

ACTOR_CRITIC_FILE = "ppo_actor_critic.pth"
CONFIG_FILE = "config.yaml"


class PPOEnsembleAgent(NamedAgent):
    def __init__(self, cfg: OmegaConf, agents: list[NamedAgent], name_suffix: str = ""):
        super().__init__(name=f"PPOEnsemble{name_suffix}")
        self.cfg = cfg
        self.agents = agents
        self.ppo = PPO(
            cfg=cfg,
            observation_dim=cfg.env_observation_dim + 2 * len(agents),
            num_different_actions=len(agents),
        )

        self.agent_repeat = cfg.agent_repeat
        self.same_agent_counter = self.agent_repeat
        self.last_ppo_action = None

    def on_start_game(self, game_id):
        self.same_agent_counter = self.agent_repeat
        self.last_ppo_action = None

    def get_agent_actions(self, obs: np.ndarray) -> list[np.ndarray]:
        """Gets actions from all ensemble agents for the given observation."""
        agent_actions = []
        for agent in self.agents:
            action = agent.get_step(obs)
            agent_actions.append(action)
        return agent_actions

    def extend_state_with_q_values(
        self, obs: np.ndarray, agent_actions: list[np.ndarray]
    ) -> np.ndarray:
        """Extends the state representation with Q-values from all agents."""
        extended_obs = obs.copy()
        for agent, action in zip(self.agents, agent_actions):
            q_values = agent.QValues(obs, action)
            extended_obs = np.concatenate([extended_obs, q_values])
        return extended_obs

    def get_step(self, obs: np.ndarray) -> np.ndarray:
        """
        Extends state, gets discrete choice from PPO, and returns the action of the chosen agent.
        Takes care of agent repeat.
        """
        if self.same_agent_counter < self.agent_repeat:
            self.same_agent_counter += 1
            return self.agents[self.last_ppo_action].get_step(obs)

        agent_actions = self.get_agent_actions(obs)
        extended_obs = self.extend_state_with_q_values(obs, agent_actions)
        action = self.ppo.actor_critic.get_action(
            torch.Tensor(extended_obs).to(self.ppo.device)
        )
        self.last_ppo_action = action.cpu().numpy().item()
        self.same_agent_counter = 1
        return agent_actions[self.last_ppo_action]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.ppo.actor_critic.state_dict(), path / ACTOR_CRITIC_FILE)
        with open(path / CONFIG_FILE, "w") as f:
            OmegaConf.save(self.cfg, f)

    def save_to_wandb(self, wandb_run: wandb.Run, folder_name: str) -> None:
        save_dir = Path(wandb_run.dir) / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save(save_dir)

        wandb.save(str(save_dir / ACTOR_CRITIC_FILE))
        wandb.save(str(save_dir / CONFIG_FILE))
