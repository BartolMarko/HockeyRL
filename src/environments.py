from hockey.hockey_env import HockeyEnv


class SparseRewardHockeyEnv(HockeyEnv):
    """Remove reward shaping from the HockeyEnv. (Remove closeness to puck reward.)"""

    def get_reward(self, info: dict) -> float:
        return self._compute_reward()
