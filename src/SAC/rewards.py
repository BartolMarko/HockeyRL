import gymnasium as gym

class RewardShaper:
    def __init__(self, cfg):
        self.cfg = cfg

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

    def transform(self, reward, info, done_or_truncated):
        if self.cfg.reward_transform == 'v1':
            return self.transform_v1(reward, info, done_or_truncated)
        else:
            # using v0 as default
            return reward
