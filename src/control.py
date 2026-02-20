import numpy as np

from hockey import hockey_env as h_env

TIME_STEP = 1.0 / h_env.FPS
PUCK_LINEAR_DAMPING = 0.05
PLAYER_MAX_SPEED = 10.0

# From HockeyEnv.player1
PLAYER_MASS = 58.0
PLAYER_INERTIA = 4.57809


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


def simulate_player_step(player_state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """
    Approximate simulation of one step of player physics, ignoring interactions with the puck and edges.
    """
    pos_x, pos_y, angle, vel_x, vel_y, angular_vel = player_state

    velocity = np.array([vel_x, vel_y])
    speed = np.linalg.norm(velocity)
    force = np.array(action[:2], dtype=float) * h_env.FORCEMULTIPLIER

    # Adapted from _apply_translation_action_with_max_speed
    if pos_x > -h_env.ZONE:
        force[0] = 0
        if vel_x > 0:
            force[0] = -2 * vel_x * PLAYER_MASS / TIME_STEP
            force[0] -= pos_x * vel_x * PLAYER_MASS / TIME_STEP
        linear_damping = 20.0
    elif speed < PLAYER_MAX_SPEED:
        linear_damping = 5.0
    else:
        linear_damping = 20.0
        delta_velocity = TIME_STEP * force / PLAYER_MASS
        if np.linalg.norm(velocity + delta_velocity) >= speed:
            force = np.array([0.0, 0.0])

    acceleration = force / PLAYER_MASS
    new_vel = velocity + acceleration * TIME_STEP
    damping_factor = max(0.0, 1.0 - TIME_STEP * linear_damping)
    new_vel *= damping_factor
    new_pos = np.array([pos_x, pos_y]) + new_vel * TIME_STEP

    # Adapted from _apply_rotation_action_with_max_angle
    torque = float(action[2]) * h_env.TORQUEMULTIPLIER
    if abs(angle) > h_env.MAX_ANGLE:
        torque = 0
        if angle * angular_vel > 0:
            torque = -0.1 * angular_vel * PLAYER_MASS / TIME_STEP
        torque += -0.1 * angle * PLAYER_MASS / TIME_STEP
        angular_damping = 10.0
    else:
        angular_damping = 2.0

    angular_acceleration = torque / PLAYER_INERTIA
    new_angular_vel = angular_vel + angular_acceleration * TIME_STEP
    angular_damping_factor = max(0.0, 1.0 - TIME_STEP * angular_damping)
    new_angular_vel *= angular_damping_factor
    new_angle = angle + new_angular_vel * TIME_STEP

    return np.array(
        [new_pos[0], new_pos[1], new_angle, new_vel[0], new_vel[1], new_angular_vel],
        dtype=np.float32,
    )


def move_player_towards_position(
    obs: np.ndarray,
    n: int,
    target_pos_xy: np.ndarray,
    target_angle: float,
    shoot: float,
    pos_error_threshold: float | None = None,
) -> np.ndarray | None:
    player_state = obs[:6].copy()
    # (pos_x, pos_y, angle, vel_x, vel_y, angular_vel)

    base_kp_pos = 15.0
    base_kd_pos = 2.0
    base_kp_angle = 8.0
    base_kd_angle = 2.0
    epsilon = 0.05
    actions = []

    for _ in range(n):
        pos_error = target_pos_xy - player_state[:2]
        velocity = player_state[3:5]
        angle_error = target_angle - player_state[2]
        ang_vel = player_state[5]

        distance = np.linalg.norm(pos_error)
        angle_dist = abs(angle_error)

        kp_scale = 1.0 + 0.5 / (distance + epsilon)
        kd_scale = 1.0 + 0.3 / (distance + epsilon)
        kp_pos = base_kp_pos * kp_scale
        kd_pos = base_kd_pos * kd_scale

        kp_angle_scale = 1.0 + 0.3 / (angle_dist + epsilon)
        kd_angle_scale = 1.0 + 0.2 / (angle_dist + epsilon)
        kp_angle = base_kp_angle * kp_angle_scale
        kd_angle = base_kd_angle * kd_angle_scale

        action_pos = pos_error * kp_pos - velocity * kd_pos
        action_angle = angle_error * kp_angle - ang_vel * kd_angle
        action = np.array(
            [action_pos[0], action_pos[1], action_angle, shoot], dtype=np.float32
        )
        action = np.clip(action, -1.0, 1.0)
        actions.append(action)

        player_state = simulate_player_step(player_state, action)

    if (
        pos_error_threshold is not None
        and np.max(np.abs(target_pos_xy - player_state[:2])) > pos_error_threshold
    ):
        return None

    return np.array(actions)
