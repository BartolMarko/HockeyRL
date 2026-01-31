import torch
from hockey.hockey_env import HockeyEnv

from src.named_agent import NamedAgent, WeakBot, StrongBot, SACLastYearAgent
from src.TDMPC.agent import TDMPCAgent
from src.TD3.td3 import TD3
from src.TD3.config_reader import Config


def create_td3_agent(name: str, weights_path: str, config_path: str) -> TD3:
    """Creates a TD3 agent for self-play evaluation."""
    env = HockeyEnv()

    td3_cfg = Config(config_path)
    td3_cfg = td3_cfg.get("td3", td3_cfg)
    TD3.enhance_cfg(td3_cfg, env)

    td3_agent = TD3(td3_cfg)
    td3_agent.restore_state(torch.load(weights_path))
    td3_agent.name = name

    return td3_agent


def agent_factory(agent_name: str, agent_cfg: dict) -> NamedAgent:
    """Factory function to create agents based on the configuration."""
    match agent_cfg["type"]:
        case "TD3":
            return create_td3_agent(
                name=agent_name,
                weights_path=agent_cfg["weights_path"],
                config_path=agent_cfg["config_path"],
            )
        case "TDMPC":
            tdmpc_agent = TDMPCAgent(
                load_dir=agent_cfg["load_dir"],
                tdmpc=None,
                step=agent_cfg.get("step", None),
                eval_mode=True,
                name_suffix="",
            )
            tdmpc_agent.name = agent_name
            return tdmpc_agent
        case "SACLastYear":
            return SACLastYearAgent(env=HockeyEnv())
        case "WeakBot":
            return WeakBot()
        case "StrongBot":
            return StrongBot()
        case _:
            raise ValueError(f"Unknown agent type: {agent_cfg['type']}")
