import numpy as np

from src.named_agent import NamedAgent
from .helper import get_my_sac


def _average(acts):
    return np.mean(list(acts.values()), axis=0), {}


def _last_man_standing(acts):
    # takes mean, removes the farthest iteratively until one is left
    meta = {}
    while len(acts) > 1:
        mean = np.mean(list(acts.values()), axis=0)
        dists = {k: np.linalg.norm(v - mean) for k, v in acts.items()}
        farthest = max(dists, key=dists.get)
        del acts[farthest]
    meta['last'] = list(acts.keys())[0]
    return list(acts.values())[0], meta


class Ensemble(NamedAgent):
    method_fns = {
        "mean": _average,
        "last-man-standing": _last_man_standing
    }

    def __init__(self, name, agent_list, method="mean"):
        super().__init__(name)
        self.agent_list = agent_list
        self.method = method
        self._method_fn = self.method_fns[method]
        self.stats = []

    @staticmethod
    def create_agent(cfg):
        agent_list = []
        for agent_cfg in cfg.agents:
            print(agent_cfg)
            cfg_path = agent_cfg.config_path
            w_path = agent_cfg.weights_folder
            agent_list.append(get_my_sac(cfg_path, w_path))
        return Ensemble("Ensemble", agent_list, method=cfg.method)

    def act(self, obs):
        acts = {}
        for agent in self.agent_list:
            act = agent.act(obs)
            acts[agent.name] = act
        act, meta = self._method_fn(acts)
        self.stats.append(meta)
        return act

    def get_step(self, obs):
        return self.act(obs)

    def show_stats(self):
        print(f"Ensemble method: {self.method}")
        if self.method == "last-man-standing":
            last_counts = {}
            for stat in self.stats:
                last = stat['last']
                last_counts[last] = last_counts.get(last, 0) + 1
            print("Last man standing counts:")
            for agent_name, count in last_counts.items():
                print(f"{agent_name}: {count}")
