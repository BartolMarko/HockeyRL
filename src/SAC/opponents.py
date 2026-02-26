import numpy as np
from comprl.client.agent import Agent
from hockey import hockey_env as h_env
from . import puffer_wrapper as pfw
from .sampler import get_sampler_using_config


class NamedAgent(Agent):
    """
    Agent with a name attribute, should be used for evaluation purposes.

    Functions that should be overriden are on_start_game, on_end_game and
    get_step.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name


class WeakBot(NamedAgent):
    def __init__(self) -> None:
        super().__init__(name="WeakBot")
        self.bot = h_env.BasicOpponent(weak=True)

    def get_step(self, obs: np.ndarray) -> np.ndarray:
        return self.bot.act(obs)


class StrongBot(NamedAgent):
    def __init__(self) -> None:
        super().__init__(name="StrongBot")
        self.bot = h_env.BasicOpponent(weak=False)

    def get_step(self, obs: np.ndarray) -> np.ndarray:
        return self.bot.act(obs)


class OpponentInPool(NamedAgent):
    def __init__(self, agent, index) -> None:
        if hasattr(agent, "name"):
            name = agent.name
        else:
            name = str(agent).replace("<", "").replace(">", "")
            name = name.replace(" ", "_")
            name = f"{name}_pool_{index}"
        super().__init__(name=name)
        self.index = index
        self.agent = agent
        self.sample_count = 0
        self.win_rate = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
        self.games_played = 0
        self.is_playable = True

        # only for SelfPlayManager
        self.pool = getattr(agent, 'pool', None)
        self.ep = getattr(agent, 'ep', None)

    def get_agent_name(self):
        return self.name

    def active(self):
        return True

    def playable(self):
        return self.is_playable

    def is_mgr(self):
        return False

    def get_agent(self):
        return self.agent

    def get_step(self, obs: np.ndarray) -> np.ndarray:
        if hasattr(self.agent, "get_step"):
            return self.agent.get_step(obs)
        elif hasattr(self.agent, "act"):
            return self.agent.act(obs)
        elif hasattr(self.agent, "plan"):
            return self.agent.plan(obs)
        else:
            raise NotImplementedError(
                    "The base agent does not have a method to get actions.")

    def get_win_rate(self):
        return self.win_rate

    def get_agent_win_rate(self):
        """ Returns the win rate of the AGENT against this opponent.
            Agent Win = Opponent Loss.
        """
        if self.games_played == 0:
            return 0.0
        total = self.win_count + self.loss_count + self.draw_count
        return self.loss_count / total

    def record_play_scores(self, win_count, loss_count, draw_count):
        self.win_count = win_count
        self.loss_count = loss_count
        self.draw_count = draw_count
        game_count = win_count + loss_count + draw_count
        if game_count == 0:
            self.win_rate = 0.0
        else:
            self.win_rate = win_count / game_count
        self.games_played += game_count

    def get_games_played(self):
        return self.games_played

    def show_scoreboard(self):
        games_played = self.get_games_played()
        win_rate = self.get_win_rate()
        difficulty_score = self.compute_difficulty_score()
        performance_score = self.compute_performance_score()
        print(f"[OPNT]: {self.name} | All Games Played: {games_played} | "
              f"Latest Wins: {self.win_count}, "
              f"Loses: {self.loss_count}, Draws: {self.draw_count}, "
              f"Win Rate: {win_rate:.2f} | "
              f"Difficulty Score: {difficulty_score:.2f} | "
              f"Performance Score: {performance_score:.2f}")

    def compute_difficulty_score(self):
        # a higher score means this opponent is difficult to beat
        if self.get_games_played() == 0:
            return 0.0
        total = self.win_count + self.loss_count + self.draw_count
        return (self.win_count + 0.5 * self.draw_count) / total

    def compute_performance_score(self):
        if self.get_games_played() == 0:
            return 0.0
        return (self.win_count - self.loss_count) / self.get_games_played()

    def show_info(self, level=1):
        print(" "*(2 * level) + f"Opponent Name: {self.name}")

    def log_stats(self, logger, episode_index):
        if logger is None:
            return
        win_rate = self.get_win_rate()
        logger.add_scalar(f"Opponent/{self.name}/win_rate", win_rate)
        logger.add_scalar(f"Opponent/{self.name}/sample_count",
                          self.sample_count)

    def plan_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        if hasattr(self.agent, "plan_batch"):
            return self.agent.plan_batch(obs_batch)
        else:
            # Fallback to single step planning
            return np.array([self.get_step(obs) for obs in obs_batch])

    def __str__(self):
        return self.name

    def __len__(self):
        # compatibility with SelfPlayManager
        return 1


class OpponentPool:
    def __init__(self, cfg=None):
        self.opponents = []
        self.last_sampled_indices = []
        self.sampler = get_sampler_using_config(cfg)
        self.name = "{}__opponent_pool".format(self.sampler.id())
        self.cfg = cfg
        self.self_play_active = False
        self.pending_activations = []
        self.num_envs = cfg.get('num_envs', 1) if cfg is not None else 1
        self.is_vec_env = self.num_envs > 1

    def add_opponent(self, opponent, priority: float = 1.0):
        if isinstance(opponent, OpponentInPool):
            if opponent.is_mgr():
                new_opponent = opponent
                new_opponent.index = len(self.opponents)
                priority = 0.0  # SelfPlayManager starts with 0 priority
            else:
                print("WARNING: Adding an OpponentInPool that is not a manager"
                      ". Creating a new OpponentInPool instead.")
                new_opponent = OpponentInPool(opponent.agent,
                                              index=len(self.opponents))
        else:
            new_opponent = OpponentInPool(opponent, index=len(self.opponents))
        self.opponents.append(new_opponent)
        self.sampler.add_arm(weight=priority)

    def sample_opponent(self, count: int = 1):
        sample_indices = self.sampler.sample(count)
        self.last_sampled_indices = sample_indices
        sampled_opponents = [self.opponents[i] for i in sample_indices]
        for opponent in sampled_opponents:
            opponent.sample_count += 1
            if opponent.is_mgr():
                # sample_opponent() in selfplaymgr assigns an agent to itself
                opponent.sample_opponent()
            assert opponent.agent is not None, \
                f"Sampled opponent {opponent.name} has no agent assigned."
        if count == 1:
            return sampled_opponents[0]
        return sampled_opponents

    def update_opponent_priority(self, index: int, new_priority: float):
        assert 0 <= index < len(self.opponents), "Invalid opponent index." + \
         f" Got {index}, but length is {len(self.opponents)}."
        self.sampler.update_arm(index, new_priority)

    def end_evaluation(self):
        """
        For now the function checks if we have a SelfPlayManager in the pool,
        and if so, does it need to be activated
        """
        for idx, opponent in enumerate(self.opponents):
            if opponent.is_mgr():
                opponent.end_evaluation(self)
                if opponent.active():
                    if len(opponent) > 0:
                        self._activate_self_play(idx)
                    else:
                        self._postpone_self_play_activation(idx)

    def get_last_eval_win_rate(self):
        """
        Used by the selfplaymgr to determine whether to activate itself
        """
        win_rate = 0.0
        if not self.get_last_opponents():
            return 0.0
        for opponent in self.get_last_opponents():
            win_rate += opponent.get_agent_win_rate()
        return win_rate / len(self.get_last_opponents())

    def is_self_play_active(self):
        return self.self_play_active

    def update_pool(self, agent, episode_index, logger=None):
        for opponent in self.opponents:
            opponent.log_stats(logger, episode_index)
            if opponent.is_mgr() and opponent.active():
                opponent.add_episode_number_to_pool(agent, episode_index)
        self._process_pending_activations()

    def _postpone_self_play_activation(self, idx: int):
        print(f"[SPLY] Postponing activation of {self.opponents[idx].name} due"
              " to empty pool.")
        self.pending_activations.append(idx)

    def _process_pending_activations(self):
        for idx, pidx in enumerate(self.pending_activations):
            if len(self.opponents[pidx]) > 0:
                self._activate_self_play(pidx)
                self.pending_activations.pop(idx)

    def _activate_self_play(self, idx: int):
        selfplaymgr = self.opponents[idx]
        if not selfplaymgr.is_mgr():
            raise ValueError(
                    "[SPLY] Provided opponent is not a SelfPlayManager but {}."
                    .format(selfplaymgr.name))
        if selfplaymgr not in self.opponents:
            raise ValueError(
                    "[SPLY] Provided SelfPlayManager {} is not in the pool."
                    .format(selfplaymgr.name))
        if len(selfplaymgr) == 0:
            raise ValueError(
                    "[SPLY] Cannot activate {} with empty pool."
                    .format(selfplaymgr.name))
        if self.sampler.get_weights()[selfplaymgr.index] > 0:
            # already active
            return

        print(f"[SPLY] Activating self-play manager {selfplaymgr.name}...")
        priority = selfplaymgr.get_priority()
        current_weights = self.sampler.get_weights()
        total_weight = sum(current_weights)

        # SelfPlay = priority, Others sum to 1-priority.
        for idx, opponent in enumerate(self.opponents):
            if opponent == selfplaymgr:
                weight = priority
            else:
                weight = (current_weights[idx] / total_weight) * (1 - priority)
            self.sampler.update_arm(idx, weight)
        self.self_play_active = True
        print("[SPLY] Self-play activated. Weights updated")
        for idx, opponent in enumerate(self.opponents):
            print(f"  {opponent.name}: {self.sampler.get_weights()[idx]:.3f}")

    def self_play_mgr_needs_save(self, agent: Agent, env_step: int) -> bool:
        for opponent in self.opponents:
            if opponent.is_mgr() and \
                    opponent.add_episode_number_to_pool(agent, env_step):
                return True
        return False

    def reset_sampler(self):
        self.sampler.reset()

    def clear_opponents(self):
        self.opponents = []
        self.sampler = get_sampler_using_config(self.cfg)

    def remove_opponent(self, index: int):
        assert 0 <= index < len(self.opponents), "Invalid opponent index." + \
         f" Got {index}, but length is {len(self.opponents)}."
        self.opponents.pop(index)
        self.sampler.remove_arm(index)

    def get_last_opponents(self):
        return [self.opponents[i] for i in self.last_sampled_indices]

    def get_all_opponents(self):
        return self.opponents

    def get_playable_opponents(self):
        return [opponent for opponent in self.opponents if opponent.playable()]

    def show_scoreboard(self):
        weights = self.sampler.get_weights()
        for idx, opponent in enumerate(self.opponents):
            opponent.show_scoreboard()
            print(f"  Sampling Weight: {weights[idx]:.3f}")

    def show_info(self):
        weights = self.sampler.get_weights()
        print(f"Opponent Pool: {self.name}")
        for idx, opponent in enumerate(self.opponents):
            print(f"- Opponent {idx}:")
            print(f"--- Weight: {weights[idx]:.3f}")
            print(opponent.show_info(level=2))

    def add_to_self_play(self, agent: Agent, episode_number: int) -> None:
        for opponent in self.opponents:
            if opponent.is_mgr() and opponent.active():
                opponent.add_episode_number_to_pool(agent, episode_number)
                print(f"[SPLY] Added episode {episode_number} to self-play "
                      f"pool. New Pool Size: {len(opponent)}")

    def __len__(self):
        return len(self.opponents)

    def __str__(self):
        return self.name

    def __add__(self, other):
        if not isinstance(other, OpponentPool):
            raise ValueError("Can only add another OpponentPool.")
        new_pool = OpponentPool()
        old_weights = self.sampler.get_weights()
        for idx, opponent in enumerate(self.opponents):
            priority = old_weights[idx]
            new_pool.add_opponent(opponent.agent, priority)
        other_weights = other.sampler.get_weights()
        for idx, opponent in enumerate(other.opponents):
            priority = other_weights[idx]
            new_pool.add_opponent(opponent.agent, priority)
        return new_pool


def get_bot_pool(cfg) -> OpponentPool:
    weak_prior = cfg.get('weak_bot_priority', 0.5)
    strg_prior = cfg.get('strg_bot_priority', 0.5)
    pool = OpponentPool()
    pool.add_opponent(WeakBot(), priority=weak_prior)
    pool.add_opponent(StrongBot(), priority=strg_prior)
    return pool


def get_opponent_pool(cfg, env=None) -> OpponentPool:
    pool = OpponentPool(cfg)
    opponents_cfg = cfg.get('opponents', [])
    for opp_cfg in opponents_cfg:
        opp_type = opp_cfg.get('type', 'WeakBot')
        priority = opp_cfg.get('priority', 1.0)
        if opp_type == 'WeakBot':
            if pool.is_vec_env:
                opponent = pfw.VecWeakBot(cfg.num_envs)
            else:
                opponent = WeakBot()
        elif opp_type == 'StrongBot':
            if pool.is_vec_env:
                opponent = pfw.VecStrongBot(cfg.num_envs)
            else:
                opponent = StrongBot()
        elif opp_type == 'PuckFollowBot':
            from . import adversarial as adv
            opponent = adv.create_puck_follow_bot(cfg.num_envs)
        elif opp_type == 'CustomAgent':
            experiment_name = opp_cfg.get('experiment_name')
            from .helper import load_agent_from_config
            opponent = load_agent_from_config(experiment_name, env)
        elif opp_type == 'CustomPool':
            # this adds a pool of bots which get activated after set iters
            from .selfplaymgr import create_selfplay_manager
            opponent = create_selfplay_manager(cfg, opp_cfg)
        else:
            raise ValueError(f"Unknown opponent type: {opp_type}")
        pool.add_opponent(opponent, priority=priority)

    # Add Self-Play Manager if configured
    if 'self_play' in cfg:
        from .selfplaymgr import create_selfplay_manager
        for sp_cfg in cfg.self_play:
            pool.add_opponent(create_selfplay_manager(cfg, sp_cfg))
    return pool


def test_opponent_pool():
    pool = OpponentPool()
    pool.add_opponent(WeakBot(), priority=0.7)
    pool.add_opponent(StrongBot(), priority=0.3)

    for _ in range(10):
        opponent = pool.sample_opponent()
        print(f"Sampled Opponent: {opponent.name}")

    pool.update_opponent_priority(0, 0.2)
    pool.update_opponent_priority(1, 0.8)

    pool.show_scoreboard()


def test_opponent_pool_add():
    pool1 = OpponentPool()
    pool1.add_opponent(WeakBot(), priority=0.6)

    pool2 = OpponentPool()
    pool2.add_opponent(StrongBot(), priority=0.4)

    combined_pool = pool1 + pool2
    combined_pool.show_scoreboard()


def verify_sampling_distribution(pool: OpponentPool, num_samples: int = 10000):
    sample_counts = {opponent.name: 0 for opponent in pool.get_all_opponents()}
    for _ in range(num_samples):
        opponent = pool.sample_opponent()
        sample_counts[opponent.name] += 1

    sampling_weights = pool.sampler.get_weights()
    total_weight = sum(sampling_weights)
    expected_distribution = {opponent.name: weight / total_weight
                             for opponent, weight in zip(
                                 pool.get_all_opponents(), sampling_weights)}

    print(f"Sampling Distribution {pool} after", num_samples, "samples:")
    for opponent_name, count in sample_counts.items():
        empirical_prob = count / num_samples
        expected_prob = expected_distribution[opponent_name]
        bound = np.sqrt((expected_prob * (1 - expected_prob)) / num_samples)
        confidence_interval = 1.96 * bound
        print(f"Opponent: {opponent_name}, Empirical: {empirical_prob:.4f}, "
              f"Expected: {expected_prob:.4f}, CI: ±{confidence_interval:.4f}")


if __name__ == "__main__":
    # test_opponent_pool()
    # test_opponent_pool_add()
    # delta_uniform
    cfg = {'opponent_pool': {'type': 'delta_uniform'}}
    cfg['opponents'] = [
        {'type': 'WeakBot', 'priority': 0.7},
        {'type': 'StrongBot', 'priority': 0.3}
    ]
    pool = get_opponent_pool(cfg)
    verify_sampling_distribution(pool)
    # pool = get_bot_pool({'weak_bot_priority': 1, 'strg_bot_priority': 0})
    # verify_sampling_distribution(pool)
    # pool = get_bot_pool({'weak_bot_priority': 0, 'strg_bot_priority': 1})
    # verify_sampling_distribution(pool)
    print("All tests passed.")
