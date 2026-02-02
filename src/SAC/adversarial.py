import numpy as np

class PuckFollowBot:
    def __init__(self, name):
        self.name = name

    def act(self, obs):
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
        # Constants
        GOAL_X = -3.6
        kP = 20.0
        kD = 2

        p1 = np.asarray([obs[0], obs[1], obs[2]])
        v1 = np.asarray(obs[3:6])
        puck = np.asarray(obs[12:14])
        puckv = np.asarray(obs[14:16])

        def puck_moving_towards_our_goal(puck_pos, puck_vel):
            # Our goal is at negative x
            return puck_vel[0] < 0 and puck_pos[0] < 0

        if puck[0] < 0:
            # Attack Mode for puck on our side
            target_pos = puck[0:2]
            target_pos += puckv * 0.1
        else:
            # Defend Mode, go back to goal line, align with puck y
            target_y = puck[1] + puckv[1] * 0.2
            target_y = np.clip(target_y, -1.25, 1.25)
            target_pos = np.array([GOAL_X, target_y])

        # Face the puck/target
        target_angle = np.arctan2(puck[1] - p1[1], puck[0] - p1[0])

        # Compute action
        action = np.zeros(4, dtype=np.float32)

        # Position Control (PD Controller) - thanks electronics degree
        # k_p for pos = 10 (approx), k_d for vel = 0.5
        action[0] = (target_pos[0] - p1[0]) * kP- v1[0] * kD
        action[1] = (target_pos[1] - p1[1]) * kP - v1[1] * kD
        action[2] = ((target_angle - p1[2] + np.pi) % (2 * np.pi) - np.pi) * 5.0 - v1[2] * 0.5
        # Always shoot
        action[3] = 1.0

        action = np.clip(action, -1.0, 1.0)
        return action

class VecPuckFollowBot:
    def __init__(self, num_envs, name="PuckFollowBot"):
        self.num_envs = num_envs
        self.opponents = [PuckFollowBot(name + "_{}".format(i)) for i in range(num_envs)]

    def plan_batch(self, obs):
        actions = []
        for i in range(self.num_envs):
            action = self.opponents[i].act(obs[i])
            actions.append(action)
        return np.array(actions)

    def act(self, obs):
        return self.plan_batch(obs)

def create_puck_follow_bot(num_envs=1, name='PuckFollowBot'):
    if num_envs == 1:
        return PuckFollowBot(name=name)
    else:
        return VecPuckFollowBot(num_envs=num_envs, name=name)

def test_bot():
    import hockey.hockey_env as h_env
    import time
    env = h_env.HockeyEnv()
    opponent = h_env.BasicOpponent(weak=False)
    bot = create_bot()
    obs, info = env.reset(one_starting=True)
    done = False
    total_reward = 0.0

    while not done:
        action = bot.act(obs)
        opponent_action = opponent.act(env.obs_agent_two())

        obs, reward, done, truncated, info = env.step(np.hstack([action, opponent_action]))
        total_reward += reward
        env.render()
        time.sleep(0.02)

        if truncated:
            done = True

    print("Episode finished with reward:", total_reward)
    env.close()
    return total_reward

if __name__ == "__main__":
    test_bot()
