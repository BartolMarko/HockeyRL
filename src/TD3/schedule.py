from abc import ABC, abstractmethod


class Scheduler(ABC):
    @abstractmethod
    def value(self):
        ...

    def reset(self):
        pass

class LinearScheduler(Scheduler):
    def __init__(self, schedule_timesteps, final_p, init_p = 1.0):
        self.schedule_t = schedule_timesteps
        self.final_p = final_p
        self.init_p = init_p
        self.counter = 0

    def value(self):
        frac = min(float(self.counter) / self.schedule_t, 1.0)
        self.counter += 1
        return self.init_p + frac * (self.final_p - self.init_p)
    
    def reset(self):
        self.counter = 0
    
class ConstantSchedular(Scheduler):
    def __init__(self, val):
        self.val = val

    def value(self):
        return self.val
    
class SchedulerFactory:
    @staticmethod
    def get_scheduler(string):
        match string.lower():
            case 'linear':
                return LinearScheduler(1_000_000, .1, 1.0)
            case 'constant':
                return ConstantSchedular(.1)
            case _:
                return None