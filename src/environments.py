import pufferlib.vector
import numpy as np
import gymnasium

from hockey.hockey_env import HockeyEnv


class SparseRewardHockeyEnv(HockeyEnv):
    """Remove reward shaping from the HockeyEnv. (Remove closeness to puck reward.)"""

    def get_reward(self, info: dict) -> float:
        return self._compute_reward()


def environment_factory(env_name: str) -> HockeyEnv:
    """Factory function to create environment instances based on the name."""
    match env_name:
        case "HockeyEnv":
            return HockeyEnv()
        case "SparseRewardHockeyEnv":
            return SparseRewardHockeyEnv()
        case _:
            raise ValueError(f"Unknown environment name: {env_name}")


class PufferHockeyEnv(pufferlib.PufferEnv):
    def __init__(
        self, buf=None, seed=None, env_name="HockeyEnv", first_posession_left=True
    ):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(18 * 2,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(4 * 2,), dtype=np.float32
        )
        self.num_agents = 1

        super().__init__(buf)

        self.env = environment_factory(env_name)
        self.first_posession_left = first_posession_left

    def reset(self, seed=None):
        obs, _ = self.env.reset(seed=seed, one_starting=self.first_posession_left)
        obs_agent_two = self.env.obs_agent_two()

        self.observations[0, :] = self._create_vectorized_observation(
            obs, obs_agent_two
        )
        self.rewards[:] = 0
        self.terminals[:] = False
        self.truncations[:] = False

        return self.observations, []

    def step(self, actions):
        obs, reward, done, trunc, info = self.env.step(actions[0])
        obs_agent_two = self.env.obs_agent_two()

        self.observations[0, :] = self._create_vectorized_observation(
            obs, obs_agent_two
        )
        self.rewards[0] = reward
        self.terminals[0] = done
        self.truncations[0] = trunc

        return self.observations, self.rewards, self.terminals, self.truncations, [info]

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    @staticmethod
    def _create_vectorized_observation(
        obs_agent_one: np.ndarray, obs_agent_two: np.ndarray
    ) -> np.ndarray:
        return np.hstack([obs_agent_one, obs_agent_two]).astype(np.float32)


def get_per_player_observations(batched_observations: np.ndarray):
    half_dim = batched_observations.shape[1] // 2
    return batched_observations[:, :half_dim], batched_observations[:, half_dim:]


def create_vectorized_pufferlib_env(
    env_name: str,
    num_envs: int,
    seed: int,
    backend=pufferlib.vector.Multiprocessing,
    percentage_left_first: float = 0.5,
):
    """Creates a PufferLib environment wrapper for the specified environment."""
    left_first_num_envs = int(num_envs * percentage_left_first)
    right_first_num_envs = num_envs - left_first_num_envs
    env_kwargs = [
        {"env_name": env_name, "first_posession_left": True}
        for _ in range(left_first_num_envs)
    ]
    env_kwargs += [
        {"env_name": env_name, "first_posession_left": False}
        for _ in range(right_first_num_envs)
    ]

    pufferlib_env = pufferlib.vector.make(
        [PufferHockeyEnv] * num_envs,
        num_envs=num_envs,
        backend=backend,
        seed=[seed + i for i in range(num_envs)],
        env_args=[()] * num_envs,
        env_kwargs=env_kwargs,
        num_workers=num_envs,
    )
    return pufferlib_env


if __name__ == "__main__":
    vectorized_puffer_env = create_vectorized_pufferlib_env(
        "SparseRewardHockeyEnv",
        num_envs=4,
        seed=0,
        backend=pufferlib.vector.Multiprocessing,
    )
    vect_obs, _ = vectorized_puffer_env.reset()

    print(vect_obs)
    print("First player observations:")
    print("Second player observations:")

    # vect_obs, vect_rewards, vect_terminals, vect_truncations, vect_info = (
    #     vectorized_puffer_env.step(np.zeros((4, 8), dtype=np.float32))
    # )
    # print(vect_obs)
    # print(vect_rewards)
    # print(vect_terminals)
    # print(vect_truncations)
    # print(vect_info)
