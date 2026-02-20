import torch
from hockey.hockey_env import HockeyEnv

from src.named_agent import NamedAgent, WeakBot, StrongBot, SACLastYearAgent
from src.TD3.td3 import TD3
from src.TD3.config_reader import Config
from src.SAC.helper import get_my_sac

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_td3_agent(name: str, weights_path: str, config_path: str) -> TD3:
    """Creates a TD3 agent for self-play evaluation."""
    env = HockeyEnv()

    td3_cfg = Config(config_path)
    td3_cfg = td3_cfg.get("td3", td3_cfg)
    TD3.enhance_cfg(td3_cfg, env)

    td3_agent = TD3(td3_cfg)
    td3_agent.restore_state(torch.load(weights_path, map_location=device))
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
            if torch.cuda.is_available():
                from src.TDMPC.agent import TDMPCAgent
            else:
                raise RuntimeError(
                    "TDMPC agent requires CUDA."
                )
            tdmpc_agent = TDMPCAgent(
                load_dir=agent_cfg["load_dir"],
                tdmpc=None,
                step=agent_cfg.get("step", None),
                eval_mode=True,
                name_suffix="",
            )
            tdmpc_agent.name = agent_name
            return tdmpc_agent
        case "SAC":
            return get_my_sac(
                cfg_path=agent_cfg["config_path"],
                w_folder=agent_cfg["weights_folder"]
            )
        case "SACLastYear":
            return SACLastYearAgent(env=HockeyEnv())
        case "WeakBot":
            return WeakBot()
        case "StrongBot":
            return StrongBot()
        case _:
            raise ValueError(f"Unknown agent type: {agent_cfg['type']}")


def test_sac_agent_factory():
    """Test function for the SAC agent factory."""
    config_yaml_path = "src/SAC/sac_example.yaml"
    import yaml
    with open(config_yaml_path, 'r') as file:
        sac_config = yaml.safe_load(file)
    sac_agent = agent_factory("TestSACAgent", sac_config)
    assert isinstance(sac_agent, NamedAgent), \
        "The created agent should be an instance of NamedAgent."
    print("SAC agent factory test passed.")


if __name__ == "__main__":
    test_sac_agent_factory()
