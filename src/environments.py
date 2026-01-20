from hockey.hockey_env import HockeyEnv


class SparseRewardHockeyEnv(HockeyEnv):
    """Remove reward shaping from the HockeyEnv. (Remove closeness to puck reward.)"""

    def get_reward(self, info: dict) -> float:
        return self._compute_reward()


def environment_factory(env_name: str):
    """Factory function to create environment instances based on the name."""
    match env_name:
        case "HockeyEnv":
            return HockeyEnv()
        case "SparseRewardHockeyEnv":
            return SparseRewardHockeyEnv()
        case _:
            raise ValueError(f"Unknown environment name: {env_name}")
