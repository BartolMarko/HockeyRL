import numpy as np

from hockey import hockey_env as h_env

TIME_STEP = 1.0 / h_env.FPS
PUCK_LINEAR_DAMPING = 0.05

PUCK_X_POS_INDEX = 12
PUCK_Y_POS_INDEX = 13
PUCK_X_VEL_INDEX = 14
PUCK_Y_VEL_INDEX = 15


def get_n_future_puck_positions(observation: np.ndarray, n: int) -> np.ndarray:
    puck_pos = observation[12:14].copy()
    puck_vel = observation[14:16].copy()

    trajectory = np.zeros((n, 2))
    for i in range(n):
        current_speed = np.linalg.norm(puck_vel)

        if current_speed > h_env.MAX_PUCK_SPEED:
            linear_damping = 10.0
        else:
            linear_damping = 0.05

        damping_factor = max(0.0, 1.0 - TIME_STEP * linear_damping)
        puck_vel *= damping_factor
        puck_pos += puck_vel * TIME_STEP

        trajectory[i] = puck_pos.copy()

    return trajectory
