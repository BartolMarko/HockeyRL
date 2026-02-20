import numpy as np
import math
from hockey import hockey_env as h_env

from src.control import move_player_towards_position, get_n_future_puck_positions
from src.environments import (
    LEFT_GOAL_X,
    LEFT_GOAL_BOTTOM_CLIPPED_Y,
    LEFT_GOAL_TOP_CLIPPED_Y,
)

PLAYER_ANGLE_INDEX = 2
PLAYER_DEFENSIVE_X = LEFT_GOAL_X + 0.75
BOTTOM_TOP_MARGIN = 0.5 + 10.0 / 60.0

HAS_PUCK_1_INDEX = 16
HAS_PUCK_2_INDEX = 17


def get_puck_intersect_angle(obs: np.ndarray) -> float:
    puck_vel_x = obs[14]
    puck_vel_y = obs[15]
    return np.clip(
        math.atan2(-puck_vel_y, -puck_vel_x), -h_env.MAX_ANGLE, h_env.MAX_ANGLE
    )


def is_position_inside_player_field(pos: np.ndarray) -> bool:
    x, y = pos
    return (
        x < 0
        and x > (LEFT_GOAL_X - h_env.CENTER_X)
        and y > BOTTOM_TOP_MARGIN
        and y < h_env.H - BOTTOM_TOP_MARGIN
    )


def get_action_hints(
    obs: np.ndarray,
    horizon: int,
    num_puck_positions: int = 10,
    num_defensive_positions: int = 10,
) -> np.ndarray:
    player_angle = obs[PLAYER_ANGLE_INDEX]
    defensive_y_coordinates = np.linspace(
        LEFT_GOAL_BOTTOM_CLIPPED_Y,
        LEFT_GOAL_TOP_CLIPPED_Y,
        num_defensive_positions,
        endpoint=True,
    )
    action_sequences = [
        move_player_towards_position(
            obs,
            n=horizon,
            target_pos_xy=(PLAYER_DEFENSIVE_X - h_env.CENTER_X, y - h_env.CENTER_Y),
            target_angle=angle,
            shoot=0.0,
        )
        for y in defensive_y_coordinates
        for angle in [0.0, player_angle]
    ]
    if obs[HAS_PUCK_1_INDEX] + obs[HAS_PUCK_2_INDEX] == 0:  # if no player has the puck
        puck_future_positions = list(
            get_n_future_puck_positions(obs, num_puck_positions)
        )
        filtered_puck_future_positions = [
            pos for pos in puck_future_positions if is_position_inside_player_field(pos)
        ]
        action_sequences += [
            move_player_towards_position(
                obs,
                n=horizon,
                target_pos_xy=(pos[0] - h_env.CENTER_X, pos[1] - h_env.CENTER_Y),
                target_angle=angle,
                shoot=shoot,
            )
            for pos in filtered_puck_future_positions
            for angle in [get_puck_intersect_angle(obs), player_angle]
            for shoot in [0.0, 1.0]
        ]
    return np.stack(action_sequences, axis=1)
