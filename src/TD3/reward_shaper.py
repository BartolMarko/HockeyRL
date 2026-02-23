from abc import ABC, abstractmethod

from src.TD3.custom_opponent import ricochet_target

import numpy as np

class RewardShaper(ABC):
    @abstractmethod
    def get_reward(self, obs, action, original_reward, info):
        ...


class DefaultReward(RewardShaper):
    def get_reward(self, obs, action, original_reward, info):
        return original_reward
    


SELF_GOAL_POS = np.array([-4.25, 0.0])
MAX_TIME_STEPS = 250
class GoalGuardingReward(RewardShaper):
    MAX_DIST = np.sqrt(4.25*4.25 + 2.8 * 2.8)
    def __init__(self, max_reward = -30.0):
        self.max_reward = max_reward

    def get_reward(self, obs, action, original_reward, info):
        puck_pos_x = obs[12]
        guard_reward = 0.0
        if (puck_pos_x > 0):
            # self pos
            p1 = np.array([obs[0], obs[1]])

            diff = p1 - SELF_GOAL_POS

            # TODO: need to double check my normalization
            dist = np.sqrt(np.sum(diff * diff))
            guard_reward = dist/(self.MAX_DIST * MAX_TIME_STEPS / 2)
        return original_reward + guard_reward * self.max_reward
    
class RicochetDirectionReward(RewardShaper):
    GOAL_POS = np.array([4.25, 0.0])
    def __init__(self, max_reward = 1.5):
        self.max_reward = max_reward

    def get_reward(self, obs, action, original_reward, info):
        if action[-1] > 0.5 and obs[16] > 0:
            my_pos = np.array([obs[0], obs[1]])
            puck   = np.array([obs[12], obs[13]])
            my_angle  = obs[2]
            target_top    = ricochet_target(puck, self.GOAL_POS, 2.8)
            target_bottom = ricochet_target(puck, self.GOAL_POS, -2.8)
            if target_top is None and target_bottom is None:
                return original_reward
            elif target_top is None:
                target = target_bottom
            elif target_bottom is None:
                target = target_top
            else:
                dist_top_sq = np.sum(np.square(target_top - puck[:2]))
                dist_bottom_sq = np.sum(np.square(target_bottom - puck[:2]))
                target = target_top if dist_top_sq < dist_bottom_sq else target_bottom
            diff = target - my_pos
            target_angle = np.atan2(diff[1], diff[0])

            # map angle diff to [0, pi]
            angle_diff = abs(my_angle - target_angle)
            angle_diff = min(angle_diff, 2 * np.pi - angle_diff)

            align = 1.0 - (angle_diff / np.pi)
            
            return original_reward + (align / 25) * self.max_reward
        else:
            return original_reward

class PureRicochetReward(RewardShaper):
    GOAL_POS = np.array([4.25, 0.0])
    def __init__(self, max_reward = 1.5):
        self.max_reward = max_reward

    def get_reward(self, obs, action, original_reward, info):
        # if action[-1] > 0.5 and obs[16] > 0:
        my_pos = np.array([obs[0], obs[1]])
        puck   = np.array([obs[12], obs[13]])
        my_angle  = obs[2]
        target_top    = ricochet_target(puck, self.GOAL_POS, 2.8)
        target_bottom = ricochet_target(puck, self.GOAL_POS, -2.8)
        if target_top is None and target_bottom is None:
            # return original_reward
            target = self.GOAL_POS
        elif target_top is None:
            target = target_bottom
        elif target_bottom is None:
            target = target_top
        else:
            dist_top_sq = np.sum(np.square(target_top - puck[:2]))
            dist_bottom_sq = np.sum(np.square(target_bottom - puck[:2]))
            target = target_top if dist_top_sq < dist_bottom_sq else target_bottom
        diff = target - my_pos
        target_angle = np.atan2(diff[1], diff[0])

        # map angle diff to [0, pi]
        angle_diff = abs(my_angle - target_angle)
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)

        align = 1.0 - (angle_diff / np.pi)
        
        return (align / 25) * self.max_reward
        # else:
        #     return original_reward

class SparseDefenseReward(RewardShaper):
    def __init__(self):
        pass
    
    def get_reward(self, obs, action, original_reward, info):
        if info['winner'] == -1:
            return -10
        else: 
            return 0

class PuckClosenessDefenseReward(RewardShaper):
    def __init__(self):
        pass
    
    def get_reward(self, obs, action, original_reward, info):
        v = 0
        if info['winner'] == -1:
            v = -10
        return float(v + info['reward_closeness_to_puck'])

class RewardFactory:
    @staticmethod
    def get_reward_shaper(cfg):
        rew = cfg.get('reward_shaper', 'default')
        match rew.lower():
            case 'default':
                print("using default reward")
                return DefaultReward()
            case 'goal_guard':
                print('using guard reward')
                return GoalGuardingReward(cfg['max_reward'])
            case 'shot_pref':
                return RicochetDirectionReward(cfg['max_reward'])
            case 'sparse_defense':
                return SparseDefenseReward()
            case 'puck_closeness_defense':
                return PuckClosenessDefenseReward()
            case 'pure_bank':
                return PureRicochetReward()
