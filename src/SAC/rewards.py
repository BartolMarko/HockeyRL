import gymnasium as gym
import numpy as np

class RewardShaper:
    """
    Observation (obs):
    # 0  x pos player one
    # 1  y pos player one
    # 2  angle player one
    # 3  x vel player one
    # 4  y vel player one
    # 5  angular vel player one
    # 6  x player two
    # 7  y player two
    # 8  angle player two
    # 9 y vel player two
    # 10 y vel player two
    # 11 angular vel player two
    # 12 x pos puck
    # 13 y pos puck
    # 14 x vel puck
    # 15 y vel puck
    # Keep Puck Mode
    # 16 time left player has puck
    # 17 time left other player has puck
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.is_vec_env = cfg.get('num_envs', 1) > 1

    def transform_v1(self, reward, info, done_or_truncated):
        new_reward = 0.0
        reward_closeness_to_puck = info['reward_closeness_to_puck']
        reward_win = reward - reward_closeness_to_puck
        if (reward_win) ** 2 < 0.01:
            reward_win = -5 # negative reward for draw as well
        new_reward = reward_win * self.cfg['reward_scale']

        new_reward += reward_closeness_to_puck * self.cfg['closeness_to_puck_reward_weight']

        new_reward += info['reward_puck_direction'] * self.cfg['puck_direction_reward_weight']

        new_reward += info['reward_touch_puck'] * self.cfg['touch_puck_reward_weight']

        if not done_or_truncated:
            new_reward -= self.cfg['reward_step_penalty']

        return new_reward

    def transform_v2(self, reward, info, done_or_truncated):
        """ Sparse Rewards, only win/loss/draw rewards """
        reward_closeness_to_puck = info['reward_closeness_to_puck']
        return reward - reward_closeness_to_puck

    def transform_v3(self, reward, info, done_or_truncated, obs):
        """ Penalise when the puck is still in our court
            and when the game ends in a draw
        """
        new_reward = reward
        reward_closeness_to_puck = info['reward_closeness_to_puck']
        new_reward -= reward_closeness_to_puck
        if new_reward ** 2 < 0.01 and (done_or_truncated):
           new_reward -= -self.cfg['draw_penalty'] # negative reward for draw as well
        puck_x_vel = obs[14]
        puck_y_vel = obs[15]
        puck_speed = (puck_x_vel ** 2 + puck_y_vel ** 2) ** 0.5
        puck_x = obs[12]
        half_court_x = 5
        if puck_speed < 0.01 and puck_x < half_court_x:
            new_reward -= self.cfg['still_puck_penalty']  # penalty for puck being still
        return new_reward

    def transform_v4(self, reward, info, done_or_truncated, obs):
        # dense rewards with draw penalty, still puck_penalty
        return info['reward_closeness_to_puck'] + \
                self.transform_v3(reward, info, done_or_truncated, obs)

    def transform(self, reward, info, done_or_truncated, obs=None):
        if self.cfg.reward_transform == 'v1':
            return self.transform_v1(reward, info, done_or_truncated)
        elif self.cfg.reward_transform == 'v2':
            return self.transform_v2(reward, info, done_or_truncated)
        elif self.cfg.reward_transform == 'v3':
            return self.transform_v3(reward, info, done_or_truncated, obs)
        elif self.cfg.reward_transform == 'v4':
            return self.transform_v4(reward, info, done_or_truncated, obs)
        else:
            # using v0 as default
            return reward

    def transform_batch(self, reward_batch, info_batch, done_or_truncated_batch, obs_batch=None):
        shaped_rewards = np.zeros_like(reward_batch, dtype=np.float32)
        for i in range(len(reward_batch)):
            obs = None
            if obs_batch is not None:
                obs = obs_batch[i]
            info_ = {k: v[i] for k, v in info_batch.items()}
            shaped_rewards[i] = self.transform(
                reward_batch[i],
                info_,
                done_or_truncated_batch[i],
                obs
            )
        return shaped_rewards
