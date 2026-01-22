from abc import ABC, abstractmethod
import random

from src.environments import environment_factory


# same grammar as opponent scheduler
# TODO: make it generic


class Env(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get(self):
        ...


class SingleEnv(Env):
    def __init__(self, env):
        self.env = env

    def get(self):
        return self.env


class MultiEnv(Env):
    def __init__(self, envs, probs: list[float] | str):
        self.envs = envs
        self.probs = probs

    def get(self):
        if type(self.probs) is str:
            chosen = random.choice(self.envs)
        else:
            chosen = random.choices(self.envs, weights=self.probs, k=1)[0]

        while isinstance(chosen, Env):
            chosen = chosen.get()

        return chosen


class EnvScheduler(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_env(self, t):
        ...


class LinearEnvScheduler(EnvScheduler):
    def __init__(self, phase_shifts: list[int], env_types):
        assert len(phase_shifts) == len(env_types)
        self.phase_shifts = phase_shifts
        self.env_types = env_types
        self.c_ind = 0

        assert len(phase_shifts) > 0
        assert phase_shifts[0] == 0, "phase shift list must start with 0"
        self.get_env(0)

    def env_from_code(self, code):
        # read opponent_scheduler for grammar specification
        match code[0]:
            case 'multi':
                env_list = code[1]
                probs = list(map(lambda x: x[0], env_list))
                assert abs(sum(probs) - 1.0) < 1e-6, (
                    f"The probabilities provided in {code} do not sum up to 1"
                )
                envs = list(map(lambda x: x[1], env_list))
                envs_parsed = [self.env_from_code(env) for env in envs]
                return MultiEnv(envs_parsed, probs)

            case 'multi_uniform':
                env_list = code[1]
                envs_parsed = [self.env_from_code(env) for env in env_list]
                return MultiEnv(envs_parsed, 'uniform')

            case _:
                return SingleEnv(environment_factory(code[0]))

    def _set_current_env(self):
        self.current_env = self.env_from_code(self.env_types[self.c_ind])

    def get_env(self, t):
        if self.c_ind < len(self.phase_shifts) and t >= self.phase_shifts[self.c_ind]:
            print("Env Switch to phase ", self.c_ind)
            self._set_current_env()
            self.c_ind += 1
        return self.current_env.get()

    def trigger_phase_change(self):
        if self.c_ind < len(self.phase_shifts):
            print("Env Switch to phase ", self.c_ind)
            self._set_current_env()
            self.c_ind += 1
            return True
        return False


class ConstantEnvScheduler(EnvScheduler):
    def __init__(self):
        self.env = environment_factory('normal')

    def get_env(self, t):
        return self.env


class EnviornmentSchedulerFactory:
    @staticmethod
    def get_environment_scheduler(config) -> EnvScheduler:
        match config.get('env_scheduler', {}).get('type', 'constant'):
            case "constant":
                print("using const env")
                return ConstantEnvScheduler()

            case 'linear':
                print('using linear env')
                return LinearEnvScheduler(
                    config['env_scheduler']['phase_shifts'],
                    config['env_scheduler']['env_types']
                )

            case _:
                raise ValueError(
                    f"Invalid env type {config['env_scheduler']['type']}"
                )