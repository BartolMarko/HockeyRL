import torch
import hockey.hockey_env as h_env

from src.TD3.td3 import TD3
from src.TD3.config_reader import Config
from src.named_agent import NamedAgent
from comprl.client import launch_client

ATTACK_AGENT_PATH = "./models/td3/177k/checkpoint_step_177000_model.pt"
DEFENSE_AGENT_PATH = "./models/td3/defender/checkpoint_step_8000_model.pt"
CONFIG_PATH = "./models/td3/defender/checkpoint_step_8000_config.yaml"


class IfElseBot(NamedAgent):
    def __init__(self):
        super().__init__("IfElseBot")
        env = h_env.HockeyEnv()
        cfg = Config(CONFIG_PATH)
        TD3.enhance_cfg(cfg, env)

        self.attack_agent = TD3(cfg)
        self.defend_agent = TD3(cfg)

        self.attack_agent.restore_state(
            torch.load(ATTACK_AGENT_PATH, map_location=torch.device("cpu"))
        )
        self.defend_agent.restore_state(
            torch.load(DEFENSE_AGENT_PATH, map_location=torch.device("cpu"))
        )

        self.done = False
        self.init_step = True

    def get_step(self, obs):
        if self._is_round_start(obs):
            self.done = False
            self.init_step = True
        self_has_puck = obs[16] > 0
        puck_x_pos = obs[12]
        if puck_x_pos < 0 and self.init_step:
            self.done = True
        if (not self.done) and self_has_puck:
            self.done = True
        self.init_step = False
        if not self.done:
            return self.defend_agent.act(obs).tolist()
        return self.attack_agent.act(obs).tolist()

    def act(self, obs):
        return self.get_step(obs)

    @staticmethod
    def _is_round_start(obs):
        x_pos_p1 = obs[0]
        x_pos_p2 = obs[6]
        theta_p1 = obs[2]
        theta_p2 = obs[8]
        puck_vel_x = obs[14]
        puck_vel_y = obs[15]

        return (
            x_pos_p1 == -3.0
            and x_pos_p2 == 3.0
            and theta_p1 == 0.0
            and theta_p2 == 0.0
            and puck_vel_x == 0.0
            and puck_vel_y == 0.0
        )

    def on_start_game(self, game_id) -> None:
        pass

    def on_end_game(self, result, stats):
        pass


if __name__ == "__main__":
    launch_client(lambda _: IfElseBot())
