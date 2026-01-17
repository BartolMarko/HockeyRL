from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from enum import Enum
import os
import random

import torch
import numpy as np

from hockey.hockey_env import HockeyEnv

from src.TD3.actor_critic import Actor
from src.TD3.custom_opponent import CustomOpponent
from src.TD3.td3 import TD3

from src.named_agent import WeakBot, StrongBot, NamedAgent
from src.opponent_pool import OpponentPoolThompsonSampling, DEFAULT_PRIOR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _evaluate_agent(agent, benchmark, test_env, eval_games):
    wins = 0
    for _ in range(eval_games):
        obs, _ = test_env.reset()
        done = False
        
        while not done:
            action1 = agent.act(obs)
            obs2 = test_env.obs_agent_two()
            action2 = benchmark.act(obs2)
            action = np.hstack([action1, action2])
            obs, _, done, _, info = test_env.step(action)
        
        if info['winner'] == 1:
            wins += 1
    
    win_rate = wins / eval_games
    
    return win_rate

class ActorWrapper(NamedAgent):
    def __init__(self, actor, name):
        super().__init__(name)
        self.actor = actor

    def act(self, obs):
        return self.actor.act(obs)

    def get_step(self, obs):
        return self.act(obs)


class OpponentScheduler(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_opponent(self, t):
        ...


class Opponent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get(self):
        ...


class SingleOpponent(Opponent):
    def __init__(self, opp):
        self.opp = opp

    def get(self):
        chosen = self.opp
        while isinstance(self.opp, Opponent):
            chosen = chosen.get()
        return chosen

class SamplingStrategy(Enum):
    UNIFORM     = 1
    PRIORITIZED = 2
    
class MultiOpponent(Opponent):
    def __init__(self, opps, probs : list[float] | str):
        self.opps = opps
        # to do replace str with SamplingStrategy above
        self.probs = probs # if str only uniform is allowed for now
    
    def get(self):
        if type(self.probs) is str:
            chosen = random.choice(self.opps)
        else:
            chosen = random.choices(self.opps, weights=self.probs, k=1)[0]
        
        while isinstance(chosen, Opponent): # unrolling the grammar
            chosen = chosen.get()

        return chosen


class FixedOpponentScheduler(OpponentScheduler):
    def __init__(self, max_timesteps : int, max_pool_size : int):
        self.max_timesteps = max_timesteps

        self.t1 = int(.25 * max_timesteps)
        self.t2 = int(.65 * max_timesteps)

        self.weak_opp   = WeakBot()
        self.strong_opp = StrongBot()
        self.custom_opp = CustomOpponent()
        self.test_env   = HockeyEnv()
        # self.test_env   = SparseHockeyRewardEnv()

        self.pool = deque(maxlen=max_pool_size) # so i don't have to manually pop 

        self.sec = False
        self.third = False
        self.min_winrate = .55
        self.eval_games = 10

    def get_opponent(self, t):
        if t < self.t1:
            return self.weak_opp
        
        r = random.random()
        if t < self.t2:
            if not self.sec:
                print("Switched to phase two")
                self.sec = True
            if r < .1:
                return self.weak_opp
            # elif r < .4:
            #     return self.custom_opp
            # elif r < .3 and len(self.pool) > 0:
            #     return self._choose_random_agent()
            return self.strong_opp
        else:
            if not self.third:
                print("Switched to phase 3")
                self.third = True
            if r < .1:
                return self.weak_opp
            elif r < .8 and len(self.pool) > 0:
                return self._choose_random_agent()
            return self.strong_opp
        
    def get_opponents(self):
        opps = [self.weak_opp, self.strong_opp, self.custom_opp]
        opps.extend(self.pool)
        return opps
        
    def _choose_random_agent(self):
        return random.choice(self.pool)

    def add_agent(self, agent_config, agent_name="self_agent"):
        state_dict = deepcopy(agent_config['state_dict'])

        agent = Actor(
            **{k:v for k, v in agent_config.items() if k != 'state_dict'}
        ).to(device)

        agent.load_state_dict(state_dict)

        if _evaluate_agent(agent, self.weak_opp, self.test_env, self.eval_games) > self.min_winrate:
            wrapped = ActorWrapper(agent, agent_name)
            self.pool.append(wrapped)


class LinearOpponentScheduler(OpponentScheduler):


    def __init__(self, phase_shifts : list[int], opponent_types : list[tuple[str, None | str]], 
                 max_pool_size : int, obs_space, act_space, td3_cfg,
                 min_winrate = .55, eval_games = 20, on_phase_change = None):
        
        assert len(phase_shifts) == len(opponent_types)
        self.phase_shifts     = phase_shifts
        self.opponent_types   = opponent_types
        self.c_ind            = 0
        self.pool             = deque(maxlen=max_pool_size)
        self.min_winrate       = min_winrate
        self.eval_games       = eval_games
        self.test_env         = HockeyEnv()
        self.weak_opp         = WeakBot()
        self.on_phase_change  = on_phase_change
        self.obs_space = obs_space
        self.act_space = act_space
        self.td3_cfg = td3_cfg

        assert len(phase_shifts) > 0
        assert phase_shifts[0] == 0, "phase shift list must start with 0"
        self.get_opponent(0)

    def opponent_from_code(self, code):
        '''
        simple grammar:
        Op ::= 
        ["basic"] | ["strong"] | ["custom"] | ["self"] | ["path/to/model"]
        ["multi", [(p, Op)+]] | ["mulit_uniform", [(Op)+]]
        where p \in [0, 1] and sum(p_i) = 1
        '''
        match (code[0]):
            case 'basic': # basic opponent
                opponent = SingleOpponent(WeakBot())
            case 'strong': # strong opponent
                opponent = SingleOpponent(StrongBot())
            case 'custom': # custom opponnent
                opponent = SingleOpponent(CustomOpponent())
            case 'self': # td3 opponent
                if len(self.pool) == 0:
                    raise Exception("Need to sample from pool but is empty")
                opponent = MultiOpponent(self.pool, 'uniform')
            case 'multi': # multiple opponents #['multi', [(.3, 'basic'), (.7, 'strong')]]
                opp_list = code[1]
                probs = list(map(lambda x : x[0], opp_list))
                assert abs(sum(probs) - 1.0) < 1e-6, f"The probabilities provided in {code} do not sum up to 1"
                opps  = list(map(lambda x : x[1], opp_list))
                opps_parsed = [self.opponent_from_code(opp) for opp in opps]
                # print(opps_parsed, probs)
                return MultiOpponent(opps_parsed, probs)
            case 'multi_uniform':
                opp_list = code[1]
                opps_parsed = [self.opponent_from_code(opp) for opp in opp_list]
                return MultiOpponent(opps_parsed, 'uniform')
            case _:
                if os.path.exists(code[0]):
                    print("loading", code[0])
                    td3 = TD3(self.obs_space, self.act_space, self.td3_cfg)
                    td3.restore_state(torch.load(code[0]))
                    actor = td3.model.actor
                    actor.eval()
                    wrapped = ActorWrapper(actor)
                    return SingleOpponent(wrapped)
                raise Exception(f"Inavlid code at position 0 {code[0]}, code recevied: {code}")

        return opponent
    
    def _set_current_oppoennt(self):
        self.current_opponent = self.opponent_from_code(self.opponent_types[self.c_ind])

    # note that this class is more like a generator
    def get_opponent(self, t):
        if self.c_ind < len(self.phase_shifts) and t >= self.phase_shifts[self.c_ind]:
            print("Switch to phase ", self.c_ind)
            self._set_current_oppoennt()
            self.c_ind += 1
            self.on_phase_change()
        return self.current_opponent.get()
    
    def get_opponents(self):
        opps = [self.weak_opp, self.strong_opp, self.custom_opp]
        opps.extend(self.pool)
        return opps
    
    def trigger_phase_change(self):
        if self.c_ind < len(self.phase_shifts):
            print("Switch to phase ", self.c_ind)
            self._set_current_oppoennt()
            self.c_ind += 1
            self.on_phase_change()
            return True
        return False
    
    def add_agent(self, agent_config, agent_name = "self_agent"):
        state_dict = deepcopy(agent_config['state_dict'])

        agent = Actor(
            **{k:v for k, v in agent_config.items() if k != 'state_dict'}
        ).to(device)

        agent.load_state_dict(state_dict)

        if _evaluate_agent(agent, self.weak_opp, self.test_env, self.eval_games) > self.min_winrate:
            wrapped = ActorWrapper(agent, agent_name)
            self.pool.append(wrapped)


'''
ACHTUNG! Not tested yet.
'''
class ThompsonScheduler(OpponentPoolThompsonSampling):
    def __init__(self):
        super().__init__(opponents=[WeakBot(), StrongBot(), CustomOpponent()])

    def add_agent(self, agent_config, agent_name = "self_agent", prior = DEFAULT_PRIOR):
        state_dict = deepcopy(agent_config['state_dict'])
        agent = Actor(
            **{k:v for k, v in agent_config.items() if k != 'state_dict'}
        ).to(device)

        agent.load_state_dict(state_dict)

        wrapped = ActorWrapper(agent, agent_name)

        super().add_opponent(wrapped, prior)

    def get_opponent(self):
        return super().sample_opponent()



class OpponentSchedulerFactory:
    @staticmethod
    def get_scheduler(full_config : dict, obs_space, action_space, on_phase_change = None) -> Opponent:
        td3_config = full_config['td3']
        config = full_config['training']
        opp_cfg = config['opponent_scheduler']
        scheduler = opp_cfg['type']
        match scheduler.lower():
            case "fixed":
                return FixedOpponentScheduler(config['total_timesteps'], 
                                              max_pool_size=opp_cfg['max_pool_size'])
            case "linear":
                return LinearOpponentScheduler(opp_cfg['phase_shifts'],
                                               opp_cfg['opponent_types'],
                                               opp_cfg['max_pool_size'],
                                               min_winrate=opp_cfg['min_winrate'],
                                               eval_games=opp_cfg['eval_games'],
                                               on_phase_change = on_phase_change, 
                                               obs_space=obs_space,
                                               act_space=action_space,
                                               td3_cfg=td3_config)
            case _:
                raise Exception(f"Invalid opponent scheduler {scheduler}")
