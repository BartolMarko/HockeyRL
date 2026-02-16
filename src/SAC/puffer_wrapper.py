import time
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
import numpy as np
import hockey.hockey_env as h_env
from multiprocessing import set_start_method


class VecBasicOpponent:
    def __init__(self, num_envs, weak=False, keep_mode=True):
        self.num_envs = num_envs
        self.opponents = [
                h_env.BasicOpponent(weak=weak, keep_mode=keep_mode)
                for _ in range(num_envs)
        ]

    def plan_batch(self, obs):
        actions = []
        for i in range(self.num_envs):
            action = self.opponents[i].act(obs[i])
            actions.append(action)
        return np.array(actions)

    def act(self, obs):
        return self.plan_batch(obs)


class VecStrongBot(VecBasicOpponent):
    def __init__(self, num_envs, name='StrongBot'):
        super().__init__(num_envs, weak=False, keep_mode=True)
        self.name = name


class VecWeakBot(VecBasicOpponent):
    def __init__(self, num_envs, name='WeakBot'):
        super().__init__(num_envs, weak=True, keep_mode=True)
        self.name = name


def make_hockey_env(env_args=None, env_kwargs=None):
    return h_env.HockeyEnv()


class Float32Wrapper(gym.Wrapper):
    def reset(self, **kwargs):
        one_starting = kwargs.pop('one_starting', None)
        if one_starting is None:
            one_starting = np.random.choice([True, False])
        kwargs['one_starting'] = one_starting
        obs, info = self.env.reset(**kwargs)
        info['obs_agent_two'] = self.obs_agent_two()
        return obs.astype(np.float32), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        info['obs_agent_two'] = self.obs_agent_two()
        return obs.astype(np.float32), reward, done, truncated, info

    def obs_agent_two(self):
        obs = self.env.obs_agent_two()
        return obs.astype(np.float32)


class ShakyObservationWrapper(gym.Wrapper):
    """
    aims to have domain randomization by adding tiny bit of noise to the
    observation.
    should encourage the agent to be more reactive rather than memorize
    lets see
    """
    _STD_DEV = 0.01

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.add_noise(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return self.add_noise(obs), reward, done, truncated, info

    def add_noise(self, obs):
        # use Opponent (6,7), Puck Pos (12,13), Puck Vel (14,15)
        noise = np.random.normal(0, self._STD_DEV, size=obs.shape)
        mask = np.zeros(obs.shape)
        mask[[6, 7, 12, 13, 14, 15]] = 1
        noise = noise * mask
        return obs + noise


def wrapped_creator(*args, **kwargs):
    env = h_env.HockeyEnv()
    env = ShakyObservationWrapper(env)
    env = Float32Wrapper(env)
    return env


def wrapped_creator_test(*args, **kwargs):
    env = h_env.HockeyEnv()
    env = Float32Wrapper(env)
    return env


class HockeyVecEnv:
    def __init__(self, vec_env):
        self.vec_env = vec_env
        self._last_obs_agent_two = None
        self.num_envs = vec_env.num_envs

    def __getattr__(self, name):
        return getattr(self.vec_env, name)

    def reset(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 10000)
        # vec-env limits arguments to reset, can't use any other than seed
        obs, info = self.vec_env.reset(seed=seed)
        if isinstance(info, list) and len(info) > 0 and \
                'obs_agent_two' in info[0]:
            self._last_obs_agent_two = np.stack(
                    [i['obs_agent_two'] for i in info])
        elif isinstance(info, dict) and 'obs_agent_two' in info:
            self._last_obs_agent_two = info['obs_agent_two']
        return obs, info

    def step(self, actions):
        obs, rewards, dones, truncateds, infos = self.vec_env.step(actions)

        if isinstance(infos, dict) and 'obs_agent_two' in infos:
            self._last_obs_agent_two = infos['obs_agent_two']
        elif isinstance(infos, list):
            # Fallback if it returns list of dicts
            if len(infos) > 0 and 'obs_agent_two' in infos[0]:
                self._last_obs_agent_two = np.stack(
                        [i['obs_agent_two'] for i in infos])

        return obs, rewards, dones, truncateds, infos

    def obs_agent_two(self):
        return self._last_obs_agent_two

    def close(self):
        self.vec_env.close()

    def render(self, mode):
        return self.vec_env.render(mode)


def create_vec_env(backend, num_envs=4, eval=True):
    if eval:
        wrapped_creator_fn = wrapped_creator_test
    else:
        wrapped_creator_fn = wrapped_creator
    env_creators = [wrapped_creator_fn for _ in range(num_envs)]
    if backend == 'multiprocessing':
        backend_cls = AsyncVectorEnv
    elif backend == 'serial':
        backend_cls = SyncVectorEnv
    else:
        raise ValueError(f"Unknown backend: {backend}")

    vec_env = backend_cls(
        env_fns=env_creators
    )
    return HockeyVecEnv(vec_env)


def test_env(backend):
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    num_envs = 8
    print(f"Setting up PufferLib vectorized ({backend}) environment...")
    vec_env = create_vec_env(backend, num_envs=num_envs)

    print("Starting rollout...")
    obs, _ = vec_env.reset()

    dummy_env = make_hockey_env()
    act_space = dummy_env.action_space
    dummy_env.close()

    num_steps = 200
    for step in range(num_steps):
        actions = np.array([act_space.sample() for _ in range(num_envs)])
        next_obs, rewards, dones, truncateds, infos = vec_env.step(actions)
        _ = np.logical_or(dones, truncateds)
        _ = next_obs
        if step % 50 == 0:
            print(f"Step {step}")

    vec_env.close()


def time_fn(fn, *args, **kwargs):
    start_time = time.time()
    result = fn(*args, **kwargs)
    end_time = time.time()
    print(f"Function {fn.__name__} took {end_time - start_time:.4f} seconds")
    return result


if __name__ == "__main__":
    time_fn(test_env, 'multiprocessing')
    time_fn(test_env, 'serial')
