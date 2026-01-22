from hockey.hockey_env import HockeyEnv, Mode


class SparseRewardHockeyEnv(HockeyEnv):
    """Remove reward shaping from the HockeyEnv. (Remove closeness to puck reward.)"""

    def get_reward(self, info: dict) -> float:
        return self._compute_reward()
    
class DefenseModeEnv(HockeyEnv):
    def __init__(self,):
        super().__init__(mode=Mode.TRAIN_DEFENSE)

    def reset(self, *args, **kwargs):
        ret = super().reset(*args, **kwargs)
        self.max_timesteps = 250
        return ret

class AttackModeEnv(HockeyEnv):
    def __init__(self):
        super().__init__(mode=Mode.TRAIN_SHOOTING)

    def reset(self, *args, **kwargs):
        ret = super().reset(*args, **kwargs)
        self.max_timesteps = 250
        return ret


def environment_factory(env_name: str):
    """Factory function to create environment instances based on the name."""
    match env_name.lower():
        case "hockeyenv" | "normal":
            return HockeyEnv()
        case "sparserewardhockeyenv":
            return SparseRewardHockeyEnv()
        case "attack":
            return AttackModeEnv()
        case "defense":
            return DefenseModeEnv()
        case _:
            raise ValueError(f"Unknown environment name: {env_name}")
