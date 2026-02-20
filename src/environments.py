import numpy as np
from hockey import hockey_env as h_env
from types import MethodType
from time import sleep
import math

from src.control import (
    get_n_future_puck_positions,
    simulate_player_step,
    move_player_towards_position,
)

LEFT_GOAL_X = h_env.W / 2 - 245 / h_env.SCALE
LEFT_GOAL_CENTER_Y = h_env.H / 2
GOAL_Y_OFFSET = h_env.GOAL_SIZE / h_env.SCALE

LEFT_GOAL_TOP_Y = LEFT_GOAL_CENTER_Y + 1.0 * GOAL_Y_OFFSET
LEFT_GOAL_BOTTOM_Y = LEFT_GOAL_CENTER_Y - 1.0 * GOAL_Y_OFFSET
LEFT_GOAL_TOP_CLIPPED_Y = LEFT_GOAL_CENTER_Y + 0.9 * GOAL_Y_OFFSET
LEFT_GOAL_BOTTOM_CLIPPED_Y = LEFT_GOAL_CENTER_Y - 0.9 * GOAL_Y_OFFSET

# Test consts
PLAYER_SPAWN_POS = (LEFT_GOAL_X + 0.75, LEFT_GOAL_TOP_CLIPPED_Y)
PLAYER_TARGET_POS = (LEFT_GOAL_X + 0.75, LEFT_GOAL_BOTTOM_Y)


class PuckTestingEnv(h_env.HockeyEnv):
    def step(self, action):
        action = np.zeros_like(action)
        if self.future_action_list:
            action[:4] = self.future_action_list.pop(0)

        ret = super().step(action)
        obs = self._get_obs()

        if self.t0:
            self.drawlist = self.drawlist[:-1]
            n = 10
            n_future_positions = get_n_future_puck_positions(obs, n=n)
            for i, pos in enumerate(n_future_positions[:-2]):
                future_puck = self._create_puck(
                    (pos[0] + h_env.CENTER_X, pos[1] + h_env.CENTER_Y), (255, 0, 255)
                )
                self.drawlist.append(future_puck)
            self.drawlist.append(self.puck)
            self.world.DestroyBody(self.player1)
            self.player1 = self._create_player(
                (h_env.W / 5, LEFT_GOAL_CENTER_Y), (255, 0, 0), False
            )
            self.drawlist.append(self.player1)
            obs = self._get_obs()
            self.future_action_list = list(
                move_player_towards_position(
                    obs,
                    n,
                    n_future_positions[-1],
                    target_angle=math.atan2(-obs[15], -obs[14]),
                    shoot=0.0,
                )
            )

        self.t0 = 0
        sleep(0.5)
        return ret

    def reset(self, *args, **kwargs):
        _ = super().reset(*args, **kwargs)
        self.max_timesteps = 17
        self.t0 = 1
        self.future_action_list = []

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
        self.drawlist.extend([self.puck])

        obs = self._get_obs()
        info = self._get_info()
        return obs, info


class PlayerPredictionTestingEnv(h_env.HockeyEnv):
    def step(self, action):
        action[4:] = 0.0  # No action for player2
        old_player_state = self._get_obs()[:6].copy()
        ret = super().step(action)
        new_player_state = self._get_obs()[:6].copy()
        predicted_state = simulate_player_step(old_player_state, action[:4])
        print(
            f"Actual player state: {new_player_state}",
            f"Predicted player state: {predicted_state},",
            f"difference: {new_player_state - predicted_state},",
            f"Max difference: {np.max(np.abs(new_player_state - predicted_state))}",
            sep="\n",
        )

        sleep(1.5)
        return ret

    def reset(self, *args, **kwargs):
        _ = super().reset(one_starting=False)
        self.t0 = True
        return self._get_obs(), self._get_info()


class PlayerMovingTestingEnv(h_env.HockeyEnv):
    def step(self, action):
        action[4:] = 0.0
        if self.actions_list:
            action[:4] = self.actions_list.pop(0)
        sleep(1.5)
        return super().step(action)

    def reset(self, *args, **kwargs):
        _ = super().reset(one_starting=False)
        self.max_timesteps = 20

        self.world.DestroyBody(self.player1)
        self.world.DestroyBody(self.puck)
        self.drawlist = self.drawlist[:-3]

        # Puck should not interfere
        self.puck = self._create_puck(
            (
                5 * h_env.W / 6,
                h_env.H / 2 + self.r_uniform(-h_env.H / 2, h_env.H / 2) * 0.8,
            ),
            (0, 0, 0),
        )
        self.player1 = self._create_player(PLAYER_SPAWN_POS, (255, 0, 0), False)
        self.actions_list = list(
            move_player_towards_position(
                self._get_obs(),
                n=15,
                target_pos_xy=(
                    PLAYER_TARGET_POS[0] - h_env.CENTER_X,
                    PLAYER_TARGET_POS[1] - h_env.CENTER_Y,
                ),
                target_angle=0.0,
                shoot=0.0,
            )
        )
        self.drawlist.extend([self.player1, self.player2, self.puck])
        return self._get_obs(), self._get_info()


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
