import numpy as np
from . import helper
from . import opponents
from .sampler import get_sampler_by_name
import time
from pathlib import Path


BASE_PATH = Path(__file__).parent


class PoolingStrategy:
    """
    Base class for pooling strategies.
    """

    def __init__(self, name="pooling_strategy"):
        self.name = name

    def valid_addition(self, *args, **kwargs):
        raise NotImplementedError(
                "valid_addition method must be implemented by subclasses.")


class EpisodePoolingStrategy(PoolingStrategy):
    """
    Pools based on episode numbers.
    """

    def __init__(self, freq, name="episode_pooling_strategy"):
        super().__init__(name)
        self.freq = freq

    def valid_addition(self, *args, **kwargs):
        episode_number = kwargs['episode_number']
        return episode_number % self.freq == 0

    def __str__(self):
        return f"EpisodePoolingStrategy(freq={self.freq})"


class LastCloneWinPoolingStrategy(PoolingStrategy):
    """
    Pools based on last clone win rate.
    """

    def __init__(self, win_rate_threshold, name="win_pooling_strategy"):
        super().__init__(name)
        self.win_rate_threshold = win_rate_threshold
        self.first_agent = True

    def valid_addition(self, *args, **kwargs):
        agent_win_rate = kwargs['win_rate']
        if self.first_agent:
            # For now, the pool already has one agent when this is called
            # but its okay to have two ig?
            self.first_agent = False
            return True
        allow = agent_win_rate >= self.win_rate_threshold
        print(f"[SPLY] LastCloneWinPoolingStrategy: Agent win rate = {agent_win_rate},"
              f" Threshold = {self.win_rate_threshold} => Allow addition: {allow}")
        return allow

    def __str__(self):
        return f"LastCloneWinPoolingStrategy(delta={self.win_rate_threshold})"


def get_pooling_strategy_by_name(name, cfg):
    name = name.lower()
    if name == 'episode':
        freq = cfg.get('freq', 20)
        return EpisodePoolingStrategy(freq)
    elif name == 'all_clone_win':
        win_rate_threshold = cfg.get('win_rate_threshold', 0.8)
        return LastCloneWinPoolingStrategy(win_rate_threshold)
    else:
        raise ValueError(f"Unknown pooling strategy name: {name}")


def create_pooling_strategy(cfg):
    strategies = []
    for strat_cfg in cfg.pooling:
        strat_name = strat_cfg.get('type')
        strategy = get_pooling_strategy_by_name(strat_name, strat_cfg)
        strategies.append(strategy)
    return strategies


class SelfPlayManager(opponents.OpponentInPool):
    """
    Manages a pool of past agent versions for self-play. Keeps track of the
    episode numbers rather than agent; lol on my fear of large memory usage.
    Every sampling request loads the agent from disk. Needs to be sent to GPU
    before use.
    """

    def __init__(self, name, cfg, subcfg, index=0) -> None:
        super().__init__(None, index)
        self.name = name
        self.subcfg = subcfg
        self.pool = []
        self.pool_meta = {}
        self.cfg = cfg
        self.max_pool_size = getattr(subcfg, 'max_pool_size', np.inf)
        self.sampler = get_sampler_by_name(subcfg.sampler.name, cfg)
        self.pooling_strategies = create_pooling_strategy(subcfg)
        self.is_active_flag = False
        self.priority = 0.0
        self.current_episode = None
        self.agent = None  # will be set when sampling

    def get_priority(self):
        if self.priority == 0.0:
            self.priority = self.subcfg.get('priority', 1.0)
        return self.priority

    def playable(self):
        return self.active() and len(self.pool) > 0

    def active(self):
        return self.is_active_flag

    def is_mgr(self):
        return True

    def _check_valid_addition(self, agent, episode_number):
        if not self.active():
            return False
        if episode_number in self.pool_meta:
            # no addition of an agent played against for now
            # but might have to re-introduce them later for evals
            return False
        if len(self.pool) == 0:
            # TODO: First addition always allowed
            #   but maybe should be controlled by strategy as well
            return True
        kwargs = {
            'agent': agent,
            'episode_number': episode_number,
            'win_rate': self.get_win_rate()
        }
        for strategy in self.pooling_strategies:
            if strategy.valid_addition(**kwargs):
                return True
        return False

    def get_lowest_priority_episode(self):
        if len(self.pool) == 0:
            return None
        weights = self.sampler.get_weights()
        min_weight = min(weights)
        min_index = weights.index(min_weight)
        return self.pool[min_index]

    def add_episode_number_to_pool(self, agent, episode_number: int):
        if not self._check_valid_addition(agent, episode_number):
            return False
        if len(self.pool) >= self.max_pool_size:
            lowest_priority_episode = self.get_lowest_priority_episode()
            assert lowest_priority_episode is not None, \
                   "Lowest priority episode should not be None when pool full."
            print("[SPLY] Pool is full. Removing lowest priority episode "
                  f"{lowest_priority_episode} to add episode {episode_number}")
            idx_p = self.pool_meta[lowest_priority_episode]['index_in_pool']
            self.sampler.remove_arm(idx_p)
            self.pool.remove(lowest_priority_episode)
            del self.pool_meta[lowest_priority_episode]
            for idx, ep in enumerate(self.pool):
                self.pool_meta[ep]['index_in_pool'] = idx
        self.pool.append(episode_number)
        self.pool_meta[episode_number] = {
                'added_time': time.time(),
                'episode_number': episode_number,
                'index_in_pool': len(self.pool) - 1,
                'sample_count': 0,
                'win_count': 0,
                'loss_count': 0,
                'draw_count': 0,
                'total_games': 0
        }
        self.sampler.add_arm(weight=-1)
        sp = BASE_PATH / Path('results') / self.cfg.exp_name / 'models'
        save_path = helper.get_Nth_checkpoint(sp, episode_number)
        if agent != "test-agent":
            agent.save_models(save_path)
        return True

    def update_priorities(self, episode_number_to_score: dict):
        for episode_number, score in episode_number_to_score.items():
            if episode_number in self.pool_meta:
                index_in_pool = self.pool_meta[episode_number]['index_in_pool']
                self.sampler.update_arm(index_in_pool, score)
            else:
                print(f"WARNING! Tried to update priority for episode {episode_number},"
                      " but it's not in the pool.")

    def sample_opponent(self, count: int = 1):
        if len(self.pool) == 0:
            raise ValueError("[SPLY] Cannot sample opponent from empty pool.")
        assert count == 1, "Only single opponent sampling is supported."
        sampled_idx = self.sampler.sample(count)
        sampled_episode = self.pool[sampled_idx[0]]
        self.current_episode = sampled_episode
        if self.agent is not None:
            nth_checkpoint_dir = helper.get_Nth_checkpoint(
                    BASE_PATH / Path('results') / self.cfg.exp_name / 'models',
                    sampled_episode)
            self.agent.load_models(nth_checkpoint_dir)
        else:
            self.agent = helper.create_agent_Nth_episode(self.cfg.exp_name,
                                                         sampled_episode,
                                                         inference_only=True)
        self.agent.name = f"{self.name}_ep{sampled_episode}"
        self.agent.pool = self
        self.agent.ep = sampled_episode
        self.agent.eval()

        self.pool_meta[sampled_episode]['sample_count'] += 1
        return self.agent

    def get_last_n_opponents(self, n: int):
        n = min(n, len(self.pool))
        pool = opponents.OpponentPool()
        for episode_number in self.pool[-n:]:
            agent = helper.create_agent_Nth_episode(self.cfg.exp_name,
                                                    episode_number,
                                                    inference_only=True)
            agent.name = f"{self.name}_ep{episode_number}"
            agent.pool = self
            agent.ep = episode_number
            agent.eval()
            pool.add_opponent(agent)
        return pool

    def record_last_n_scores(self, scores: list):
        raise Exception

    def end_evaluation(self, pool):
        """
        executed at the end of evaluation, for now activates self
        if the condition is met
        """
        activation_type = self.subcfg.get('activation_type', 'always_on')
        if self.is_active_flag:
            return
        if activation_type == 'always_on':
            self.is_active_flag = True
            print(f"[SPLY] Self-Play Manager '{self.name}' set to be active (always_on)!")
        elif activation_type == 'botwin':
            win_rate = pool.get_last_eval_win_rate()
            epsilon = self.subcfg.get('activation_epsilon', 1.0)
            if win_rate >= epsilon:
                self.is_active_flag = True
                print(f"[SPLY] Self-Play Manager '{self.name}' set to be "
                      f"active win_rate {win_rate} >= {epsilon}!")
        else:
            raise ValueError(f"Unknown activation type: {activation_type}")

    def __len__(self):
        return len(self.pool)

    def stats(self):
        return {
            'pool_size': len(self.pool),
            'pool_episodes': self.pool,
            'win_rates': self.sampler.get_weights()
        }

    def record_play_scores(self, win_count, loss_count, draw_count, episode_index=None):
        if not self.active():
            return
        if episode_index is None:
            episode_index = self.current_episode
        self.win_count = win_count
        self.loss_count = loss_count
        self.draw_count = draw_count
        if episode_index not in self.pool_meta:
            print(f"[WARN] Warning: Tried to record play scores for episode {episode_index},"
                  " but it's not in the self-play pool.")
            return

        total_games = win_count + loss_count + draw_count
        self.pool_meta[episode_index].update({
            'win_rate': (win_count / total_games) if total_games > 0 else 0.0
        })
        self.pool_meta[episode_index]['total_games'] += total_games
        self.pool_meta[episode_index]['win_count'] += win_count
        self.pool_meta[episode_index]['loss_count'] += loss_count
        self.pool_meta[episode_index]['draw_count'] += draw_count
        self.sampler.update_arm(
            self.pool_meta[episode_index]['index_in_pool'],
            self.pool_meta[episode_index]['win_rate']
        )

    def log_stats(self, logger, episode_index):
        if logger is None: return
        if not self.active(): return
        logger.add_scalar(f"SelfPlay/{self.name}_pool_size", len(self.pool))
        for episode in self.pool:
            meta = self.pool_meta[episode]
            wins = meta.get('win_count', 0)
            losses = meta.get('loss_count', 0)
            draws = meta.get('draw_count', 0)
            win_rate = meta.get('win_rate', 0.0)
            logger.add_scalar(
                    f"SelfPlay_Opponents/{self.name}_ep{episode}_wins", wins)
            logger.add_scalar(
                    f"SelfPlay_Opponents/{self.name}_ep{episode}_losses",
                    losses)
            logger.add_scalar(
                    f"SelfPlay_Opponents/{self.name}_ep{episode}_draws", draws)
            logger.add_scalar(
                    f"SelfPlay_Opponents/{self.name}_ep{episode}_win_rate",
                    win_rate)
            priority = self.sampler.get_weights()[meta['index_in_pool']]
            logger.add_scalar(
                    f"SelfPlay_Opponents/{self.name}_ep{episode}_priority",
                    priority)
        # log duration in pool
        current_time = time.time()
        durations = [current_time - self.pool_meta[ep]['added_time']
                     for ep in self.pool]
        mean_duration = np.mean(durations) if len(durations) > 0 else 0.0
        max_duration = np.max(durations) if len(durations) > 0 else 0.0
        logger.add_scalar(f"SelfPlay/{self.name}_max_pool_duration",
                          max_duration)
        logger.add_scalar(f"SelfPlay/{self.name}_mean_pool_duration",
                          mean_duration)

    def show_info(self, level=1):
        prefix_space = " " * (level * 2 - 1)
        print(prefix_space + f"Self-Play Manager: {self.name}")
        print(prefix_space + f"Pool Size: {len(self.pool)}")
        if len(self.pool) > 0:
            print(prefix_space + f"Episodes in Pool: {self.pool}")
        print(prefix_space + f"Sampler: {self.sampler.name}")
        poolers = ", ".join([str(s) for s in self.pooling_strategies])
        print(prefix_space + f"Pooling Strategies: {poolers}")
        activation_type = self.subcfg.get('activation_type', 'always_on')
        print(prefix_space + f"Activation Type: {activation_type}")

    def show_scoreboard(self):
        print(f"Self-Play Manager: {self.name}")
        if len(self.pool) == 0:
            print("--- no play from pool.")
            return
        print(f"{'Episode':10} | {'Wins':5} | {'Losses':7} | {'Draws':5} | "
              f"{'Win Rate':8} | {'Priority':8} | {'Time in Pool':15}")
        print("-" * 75)
        win_rate_accum = 0.0
        total_eps = 0
        for episode in self.pool:
            meta = self.pool_meta[episode]
            wins = meta.get('win_count', 0)
            losses = meta.get('loss_count', 0)
            draws = meta.get('draw_count', 0)
            total = wins + losses + draws
            win_rate = (wins / total) if total > 0 else 0.0
            win_rate_accum += win_rate
            total_eps += total > 0
            priority = self.sampler.get_weights()[meta['index_in_pool']]
            time_in_pool = time.time() - meta['added_time']
            print(f"{episode:<10} | {wins:<5} | {losses:<7} | {draws:<5} | "
                  f"{win_rate:<8.2f} | {priority:<8.2f} | {time_in_pool:<15.2f}s")
        win_rate_avg = win_rate_accum / total_eps
        print("-" * 75)
        print(f"Average Win Rate across pool: {win_rate_avg:.2f}")
        print("-" * 75)

    def get_win_rate(self):
        win_rate = 0.0
        total_eps = 0
        for episode in self.pool:
            win_rate += self.pool_meta[episode].get('win_rate', 0.0)
            total_eps += self.pool_meta[episode].get('total_games', 0) > 1
        if total_eps > 0:
            win_rate /= len(self.pool)
            return win_rate
        elif total_eps == 0 and win_rate != 0.0:
            print("[SPLY] Warning: win_rate calculation inconsistency in "
                  f"SelfPlayManager '{self.name}'")
        return 0.0

    def get_agent_name(self):
        return f"{self.name}_ep{self.current_episode}"


def create_selfplay_manager(cfg, subcfg, name="selfplay_manager"):
    if 'name' in subcfg:
        name = subcfg.name
    return SelfPlayManager(name, cfg, subcfg)


def test_remove_lowest_priority():
    class DummySampler:
        def __init__(self):
            self.weights = []

        def add_arm(self, weight):
            self.weights.append(weight)

        def remove_arm(self, index):
            del self.weights[index]

        def get_weights(self):
            return self.weights

    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        'exp_name': 'test_exp',
        'pooling': []
    })
    subcfg = OmegaConf.create({
        'max_pool_size': 3,
        'priority': 0.8,
        'sampler': {'name': 'uniform'},
        'pooling': [{'type': 'episode', 'freq': 1}],
        'activation_type': 'always_on'
    })

    spm = SelfPlayManager("test_mgr", cfg, subcfg)
    spm.sampler = DummySampler()
    spm.is_active_flag = True

    spm.pool = [10, 20, 30]
    spm.pool_meta = {
        10: {'index_in_pool': 0},
        20: {'index_in_pool': 1},
        30: {'index_in_pool': 2}
    }
    spm.sampler.weights = [0.5, 0.2, 0.8]

    lowest_episode = spm.get_lowest_priority_episode()
    assert lowest_episode == 20, \
        f"Expected lowest priority episode to be 20, got {lowest_episode}"

    spm.add_episode_number_to_pool("test-agent", 40)
    assert 20 not in spm.pool, \
        "Episode 20 should have been removed from the pool."
    assert 40 in spm.pool, "Episode 40 should have been added to the pool."
    assert len(spm.pool) == 3, \
        f"Pool size should be 3 after addition, got {len(spm.pool)}"

    print("test_remove_lowest_priority passed.")


if __name__ == "__main__":
    test_remove_lowest_priority()
