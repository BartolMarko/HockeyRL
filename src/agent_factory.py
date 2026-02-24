import torch
from hockey.hockey_env import HockeyEnv
import os

from src.named_agent import NamedAgent, WeakBot, StrongBot, SACLastYearAgent
from src.TD3.td3 import TD3
from src.TD3.config_reader import Config
from src.SAC.helper import get_my_sac
from src import wandb_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_FOLDER = "models"


def download_agent_from_wandb(
    agent_name: str, agent_cfg: dict, save_folder: str = MODEL_SAVE_FOLDER
) -> None:
    """Downloads agent from wandb if wandb information is provided in the agent configuration."""
    if "wandb_run_id" in agent_cfg and "wandb_folder" in agent_cfg:
        wandb_utils.download_wandb_folder(
            run_id=agent_cfg["wandb_run_id"],
            wandb_folder=agent_cfg["wandb_folder"],
            destination_folder=os.path.join(save_folder, agent_name),
        )

        model_folder = os.path.join(save_folder, agent_name, agent_cfg["wandb_folder"])
        agent_cfg["load_dir"] = model_folder
        agent_cfg["weights_path"] = os.path.join(model_folder, "model.pt")
        agent_cfg["config_path"] = os.path.join(model_folder, "config.yaml")
        print(
            f"Downloaded agent: {agent_name} from run:",
            f"{agent_cfg['wandb_run_id']} folder: {agent_cfg['wandb_folder']}",
        )

    elif "wandb_artifact" in agent_cfg and agent_cfg["wandb_artifact"] is not None:
        wandb_utils.download_wandb_artifact(
            artifact_path=agent_cfg["wandb_artifact"],
            artifact_version=agent_cfg["artifact_version"],
            destination_folder=os.path.join(save_folder, agent_name),
        )

        model_folder = os.path.join(save_folder, agent_name)
        agent_cfg["weights_folder"] = model_folder
        print(
            f"Downloaded agent: {agent_name} from artifact:",
            f"{agent_cfg['wandb_artifact']}:{agent_cfg['artifact_version']}",
        )


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
    download_agent_from_wandb(agent_name, agent_cfg)
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
                cfg_path=agent_cfg["config_path"], w_folder=agent_cfg["weights_folder"]
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

    with open(config_yaml_path, "r") as file:
        sac_config = yaml.safe_load(file)
    sac_agent = agent_factory("TestSACAgent", sac_config)
    assert isinstance(sac_agent, NamedAgent), (
        "The created agent should be an instance of NamedAgent."
    )
    print("SAC agent factory test passed.")


if __name__ == "__main__":
    test_sac_agent_factory()
