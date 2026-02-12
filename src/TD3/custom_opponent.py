import random

import numpy as np

from hockey.hockey_env import SCALE, MAX_ANGLE, W, H

from src.named_agent import NamedAgent


def ricochet_target(puck, goal, wall_y):
    '''
    calculates the ricochet shots

    given the position of puck p and goal g
    need to find the target position s.t.
    when the ball hits the target position t
    it goes to the goal, it's obvious that ty \in {-Y_MAX, Y_MIN}
    because it has to be on the border, for tx
    we know that tx >= px and, and gx >= tx,
    then following ideal reflection:
    tan(theta) = (tx - px)/abs(ty - gy)
    tan(phi)   = (gx - tx)/abs(gy - ty)
    both angles must be equal so
    (tx - px)/abs(ty - gy) = (gx - tx)/abs(gy - ty)
    '''


    d1 = np.abs(goal[1] - wall_y)
    d2 = np.abs(puck[1] - wall_y)

    tx = (goal[0] * d2 + puck[0] * d1) / (d1 + d2)

    if np.abs(tx) > 4.2: # out of the board
      return None

    return np.array([tx, wall_y])


# adapted from the code of BasicOpponent
class CustomOpponent(NamedAgent):
  def __init__(self,keep_mode=True):
    super().__init__("CustomOpponent")
    self.keep_mode = keep_mode
    self.phase = np.random.uniform(0, np.pi)
    self.y_dir = random.choice([-1, 1])

  def act(self, obs, verbose=False):
    alpha = obs[2]
    # my position, theta
    p1 = np.asarray([obs[0], obs[1], alpha])
    # my velocity
    v1 = np.asarray(obs[3:6])
    # puck position
    puck = np.asarray(obs[12:14])
    # puck vel
    puckv = np.asarray(obs[14:16])

    target_pos = p1[0:2]
    target_angle = p1[2]
    
    goal_pos = np.array([4.25, 0.0])
    diff = goal_pos - p1[0:2]

    self.phase += np.random.uniform(0, 0.2)

    # time_to_break = 0.1
    # kp = 10
    # kd = 0.5

    kp = 20
    kd = 0.3
    time_to_break = 0.05

    puck_height = 30.0 / SCALE

    # if ball flies towards our goal or very slowly away: try to catch it
    if obs[16] == 0:
      # the movement only happens when the puck is not in hand
      # to make the aim more accurate
      if puckv[0] < 30.0 / SCALE:
        dist = np.sqrt(np.sum((p1[0:2] - puck) ** 2))
        # Am I behind the ball?
        if p1[0] < puck[0] and abs(p1[1] - puck[1]) < puck_height:
          # Go and kick
          target_pos = [puck[0] + 0.2, puck[1] + puckv[1] * dist * 0.1]
        else:
          # get behind the ball first
          target_pos = [-210 / SCALE, puck[1]]
      else:  # go in front of the goal
        target_pos = [-210 / SCALE, 0]
    
    if obs[16] > 0:
      # target_x = (p1[0] + goal_pos[0])/2.
      # target_y = (210 / SCALE)*self.y_dir
      # target = np.array([target_x, target_y])

      target_top    = ricochet_target(puck, goal_pos,  2.8)
      target_bottom = ricochet_target(puck, goal_pos, -2.8)

      if target_top is None and target_bottom is None:
        target = goal_pos
      elif target_top is None:
        target = target_bottom
      elif target_bottom is None:
        target = target_top
      else:
        dist_top_sq = np.sum(np.square(target_top - puck[:2]))
        dist_bottom_sq = np.sum(np.square(target_bottom - puck[:2]))
        target = target_top if dist_top_sq < dist_bottom_sq else target_bottom

      diff = target - p1[:2]
      target_angle = np.atan2(diff[1], diff[0])
    else:
      target_angle = 0.
    shoot = 0.0
    if self.keep_mode and obs[16] > 0 and obs[16] < 3:
      shoot = 1.0

    target = np.asarray([target_pos[0], target_pos[1], target_angle])
    # use PD control to get to target
    error = target - p1
    need_break = abs((error / (v1 + 0.01))) < [time_to_break, time_to_break, time_to_break * 10]
    if verbose:
      print(error, abs(error / (v1 + 0.01)), need_break)

    action = np.clip(error * [kp, kp / 5, kp / 2] - v1 * need_break * [kd, kd, kd], -1, 1)
    if self.keep_mode:
      return np.hstack([action, [shoot]])
    else:
      return action
    
  def get_step(self, obs):
    return self.act(obs)