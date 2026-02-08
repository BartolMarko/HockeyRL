import numpy as np
from hockey import hockey_env as h_env
from types import MethodType
from time import sleep

from src.control import get_n_future_puck_positions

LEFT_GOAL_X = h_env.W / 2 - 245 / h_env.SCALE
LEFT_GOAL_CENTER_Y = h_env.H / 2
GOAL_Y_OFFSET = h_env.GOAL_SIZE / h_env.SCALE

LEFT_GOAL_TOP_Y = LEFT_GOAL_CENTER_Y + 1.0 * GOAL_Y_OFFSET
LEFT_GOAL_BOTTOM_Y = LEFT_GOAL_CENTER_Y - 1.0 * GOAL_Y_OFFSET
LEFT_GOAL_TOP_CLIPPED_Y = LEFT_GOAL_CENTER_Y + 0.9 * GOAL_Y_OFFSET
LEFT_GOAL_BOTTOM_CLIPPED_Y = LEFT_GOAL_CENTER_Y - 0.9 * GOAL_Y_OFFSET


class PuckTestingEnv(h_env.HockeyEnv):
    def step(self, action):
        ret = super().step(np.zeros_like(action))
        obs = self._get_obs()

        if self.t0:
            n_future_positions = get_n_future_puck_positions(obs, n=9)
            for i, pos in enumerate(n_future_positions[:-2]):
                print(f"Future puck position {i + 1}: {pos}")
                future_puck = self._create_puck(
                    (pos[0] + h_env.CENTER_X, pos[1] + h_env.CENTER_Y), (255, 0, 255)
                )
                self.drawlist.append(future_puck)
            player = self._create_player(
                (
                    h_env.CENTER_X + n_future_positions[-1][0],
                    h_env.CENTER_Y + n_future_positions[-1][1],
                ),
                (255, 255, 0),
                False,
            )
            self.drawlist.append(player)

        self.t0 = 0
        sleep(0.5)
        return ret

    def reset(self, *args, **kwargs):
        _ = super().reset(*args, **kwargs)
        self.t0 = 1

        self.world.DestroyBody(self.player1)
        self.world.DestroyBody(self.player2)
        self.world.DestroyBody(self.puck)

        self.player1 = self._create_player((0, 0), (255, 0, 0), False)
        self.player2 = self._create_player((h_env.W, h_env.H), (0, 255, 0), True)

        self.puck = self._create_puck(
            (
                5 * h_env.W / 6,
                h_env.H / 2 + self.r_uniform(-h_env.H / 2, h_env.H / 2) * 0.8,
            ),
            (0, 0, 0),
        )

        puck_target = (LEFT_GOAL_X, LEFT_GOAL_CENTER_Y)
        puck_direction = self.puck.position - np.array(puck_target)
        puck_direction = puck_direction / puck_direction.length
        force = (
            -puck_direction
            * h_env.SHOOTFORCEMULTIPLIER
            * self.puck.mass
            / self.timeStep
        )
        self.puck.ApplyForceToCenter(force, True)

        self.drawlist = self.drawlist[:11]
        self.drawlist.extend([self.player1, self.player2, self.puck])

        obs = self._get_obs()
        info = self._get_info()
        return obs, info


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
        case "attack":
            return AttackModeEnv()
        case "defense":
            return DefenseModeEnv()
        case "defensemodeimproved":
            return DefenseModeImprovedEnv()
        case "leftfirstpossession":
            return LeftFirstPossessionEnv()
        case "rightfirstpossession":
            return RightFirstPossessionEnv()
        case _:
            raise ValueError(f"Unknown environment name: {env_name}")


def get_sparse_reward(self: h_env.HockeyEnv, info: dict) -> float:
    return self._compute_reward()


def euclidean_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def get_defense_reward_function(defense_reward_weight: float):
    def defense_reward(self: h_env.HockeyEnv, info: dict) -> float:
        win_lose_reward = self._compute_reward()

        defense_reward = 0.0
        if self.puck.position.x > h_env.CENTER_X:  # Puck is on the right half
            defense_reward -= euclidean_distance(
                (self.player1.position.x, self.player1.position.y),
                (LEFT_GOAL_X, LEFT_GOAL_CENTER_Y),
            )
        return win_lose_reward + defense_reward_weight * defense_reward

    return defense_reward


def env_reward_wrapper(
    env: h_env.HockeyEnv, reward_function_name: str, **kwargs
) -> h_env.HockeyEnv:
    """Wrap the environment to modify its reward structure."""
    match reward_function_name.lower():
        case "default" | "normal":
            return env
        case "sparse" | "sparsereward":
            env.get_reward = MethodType(get_sparse_reward, env)
            return env
        case "defense" | "defensereward":
            defense_reward_weight = kwargs["defense_reward_weight"]
            env.get_reward = MethodType(
                get_defense_reward_function(defense_reward_weight), env
            )
            return env
        case _:
            raise ValueError(f"Unknown reward function name: {reward_function_name}")
