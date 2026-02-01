import numpy as np
from hockey import hockey_env as h_env

LEFT_GOAL_X = h_env.W / 2 - 245 / h_env.SCALE
LEFT_GOAL_CENTER_Y = h_env.H / 2
GOAL_Y_OFFSET = h_env.GOAL_SIZE / h_env.SCALE

LEFT_GOAL_TOP_Y = LEFT_GOAL_CENTER_Y + 1.0 * GOAL_Y_OFFSET
LEFT_GOAL_BOTTOM_Y = LEFT_GOAL_CENTER_Y - 1.0 * GOAL_Y_OFFSET
LEFT_GOAL_TOP_CLIPPED_Y = LEFT_GOAL_CENTER_Y + 0.9 * GOAL_Y_OFFSET
LEFT_GOAL_BOTTOM_CLIPPED_Y = LEFT_GOAL_CENTER_Y - 0.9 * GOAL_Y_OFFSET


class SparseRewardHockeyEnv(h_env.HockeyEnv):
    """Remove reward shaping from the HockeyEnv. (Remove closeness to puck reward.)"""

    def get_reward(self, info: dict) -> float:
        return self._compute_reward()


class DefenseModeImprovedEnv(h_env.HockeyEnv):
    alpha = 0.4
    beta = 0.4

    left_goal_top_clipped_y = LEFT_GOAL_TOP_CLIPPED_Y
    left_goal_bottom_clipped_y = LEFT_GOAL_BOTTOM_CLIPPED_Y

    def __init__(self):
        super().__init__(mode=h_env.Mode.TRAIN_DEFENSE)

    def puck_target_y_distribution(self):
        """
        Return a number in range [-1, 1].
        Can be overriden in subclasses to change puck target distribution.
        Scaling is done so that -1 corresponds to bottom of the goal and 1 to top of the goal.
        """
        return np.random.beta(self.alpha, self.beta) * 2 - 1

    def reset(self, *args, **kwargs):
        _ = super().reset(*args, **kwargs)
        self.max_timesteps = 250
        blue = (93, 158, 199)

        self.world.DestroyBody(self.player2)
        self.world.DestroyBody(self.puck)

        self.player2 = self._create_player(
            (
                5 * h_env.W / 6 - self.r_uniform(0, h_env.W / 64),
                h_env.H / 2 + self.r_uniform(-h_env.H / 4, h_env.H / 4),
            ),
            blue,
            True,
        )

        self.puck = self._create_puck(
            (
                h_env.W / 2 + self.r_uniform(0, h_env.W / 3),
                h_env.H / 2 + self.r_uniform(-h_env.H / 2, h_env.H / 2) * 0.8,
            ),
            (0, 0, 0),
        )

        puck_target_y = (
            LEFT_GOAL_CENTER_Y + self.puck_target_y_distribution() * GOAL_Y_OFFSET
        )
        if self.puck.position.y > LEFT_GOAL_TOP_Y:
            puck_target_y = min(puck_target_y, self.left_goal_top_clipped_y)
        if self.puck.position.y < LEFT_GOAL_BOTTOM_Y:
            puck_target_y = max(puck_target_y, self.left_goal_bottom_clipped_y)

        puck_target = (LEFT_GOAL_X, puck_target_y)
        puck_direction = self.puck.position - np.array(puck_target)
        puck_direction = puck_direction / puck_direction.length
        force = (
            -puck_direction
            * h_env.SHOOTFORCEMULTIPLIER
            * self.puck.mass
            / self.timeStep
        )
        self.puck.ApplyForceToCenter(force, True)

        self.drawlist = self.drawlist[:-2]  # Remove player2 and old puck
        self.drawlist.extend([self.player2, self.puck])

        return self._get_obs(), self._get_info()


class LeftFirstPossessionEnv(h_env.HockeyEnv):
    def reset(self, *_, **__):
        ret = super().reset(one_starting=True)
        return ret


class RightFirstPossessionEnv(h_env.HockeyEnv):
    def reset(self, *_, **__):
        ret = super().reset(one_starting=False)
        return ret


class DefenseModeEnv(h_env.HockeyEnv):
    def __init__(self):
        super().__init__(mode=h_env.Mode.TRAIN_DEFENSE)

    def reset(self, *args, **kwargs):
        ret = super().reset(*args, **kwargs)
        self.max_timesteps = 250
        return ret


class AttackModeEnv(h_env.HockeyEnv):
    def __init__(self):
        super().__init__(mode=h_env.Mode.TRAIN_SHOOTING)

    def reset(self, *args, **kwargs):
        ret = super().reset(*args, **kwargs)
        self.max_timesteps = 250
        return ret


def environment_factory(env_name: str):
    """Factory function to create environment instances based on the name."""
    match env_name.lower():
        case "hockeyenv" | "normal":
            return h_env.HockeyEnv()
        case "sparserewardhockeyenv":
            return SparseRewardHockeyEnv()
        case "attack":
            return AttackModeEnv()
        case "defense":
            return DefenseModeEnv()
        case _:
            raise ValueError(f"Unknown environment name: {env_name}")
