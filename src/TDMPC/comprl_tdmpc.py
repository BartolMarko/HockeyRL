from src.TDMPC.agent import TDMPCAgent

from comprl.client import launch_client

AGENT_PATH = "./models/tdmpc2_action_hints_4_3M/checkpoint_step_4312956"


class ComprlTDMPCAgent(TDMPCAgent):
    def get_step(self, observation: list[float]) -> list[float]:
        action = super().get_step(observation).tolist()
        return action


def main() -> None:
    agent = ComprlTDMPCAgent(
        load_dir=AGENT_PATH,
        tdmpc=None,
        step=None,
        eval_mode=True,
    )
    launch_client(lambda _: agent)


if __name__ == "__main__":
    main()
