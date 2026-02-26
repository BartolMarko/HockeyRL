import argparse
import os
from PIL import Image

import torch
import numpy as np
from gymnasium import spaces

from hockey import hockey_env as h_env
from hockey.hockey_env import BasicOpponent
from src.TD3.td3 import TD3
from src.TD3.custom_opponent import CustomOpponent
from src.TD3.config_reader import Config
from src.agent_factory import WeakBot, StrongBot
from src.TD3.if_else_bot import IfElseBot

from src.episode import Episode
from src.evaluation import Heatmap

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# FOR GENERATING THE HEATMAP DIAGRAM
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, 
                        help='path for model evaluation')
    parser.add_argument('--config', type=str,
                        help='config for td3', default='./src/TD3/config.yaml')
    parser.add_argument('--opp_config', type=str,
                        help='config for td3 opp', default='./src/TD3/config.yaml')
    parser.add_argument('-m', '--maxepisodes', type=int, 
                        default=1000, help='max episodes')
    parser.add_argument('-t', '--maxtimesteps', default=2000, type=int,
                        help='max time steps/episode')
    parser.add_argument('-o', '--opponent', default='weak')
    parser.add_argument('-p', '--heatmap_path', default=".")
    parser.add_argument('-s', '--suffix', default="")
    parser.add_argument("--title", default="")
    parser.add_argument('--output', action='store',
                        default='test_output')

    return parser.parse_args()


def get_td3_cfg(cfg : Config):
    return cfg.get('td3', cfg)

def generate():

    opts = parse_args()

    cfg = get_td3_cfg(Config(opts.config))
    opp_cfg = get_td3_cfg(Config(opts.opp_config))
    print(opp_cfg)

    env = h_env.HockeyEnv()
    out_dir = opts.output

    os.makedirs(out_dir, exist_ok=True)

    opp_type = opts.opponent.lower()
    
    TD3.enhance_cfg(cfg, env)
    TD3.enhance_cfg(opp_cfg, env)
    

    match opp_type:
        case 'weak':
            agent2 = WeakBot()
        case 'strong':
            agent2 = StrongBot()
        case 'custom':
            agent2 = CustomOpponent()
        case 'if_else':
            agent2 = IfElseBot()
        case _:
            if os.path.exists(opp_type):
                agent2 = TD3(opp_cfg)
                print(agent2.model.actor)
                print("loading agent2")
                agent2.restore_state(torch.load(opp_type))
                opp_type = 'td3' # override for gif path
            else:
                raise Exception('Invalid opponent '+ opts.opponent)

    match opts.path:
        case 'weak':
            algo = WeakBot()
        case 'strong':
            algo = StrongBot()
        case 'custom':
            algo = CustomOpponent()
        case 'if_else':
            algo = IfElseBot()
        case _:
            if os.path.exists(opts.path):
                algo = TD3(opp_cfg)
                print("loading algo")
                algo.restore_state(torch.load(opts.path))
            else:
                raise Exception('Invalid opponent '+ opts.path)

    wins = 0
    max_epi = opts.maxepisodes
    max_timesteps = opts.maxtimesteps
    heatmap = Heatmap()

    for i_epi in range(1, max_epi + 1):
        ob, _info = env.reset()
        ob_agent2 = env.obs_agent_two()
        ep_reward = 0
        agent2.on_start_game(game_id=None)
        epi = Episode(ob)
        for t in range(max_timesteps):
            a1 = algo.act(ob)
            if hasattr(agent2, "act"):
                a2 = agent2.act(ob_agent2)
            else:
                a2 = agent2.get_step(ob_agent2)

            (ob_new, reward, done, trunc, info) = env.step(np.hstack([a1, a2]))

            epi.add(ob, a1, a2, reward, done or trunc)

            ob = ob_new
            ob_agent2 = env.obs_agent_two()

            ep_reward += reward

            if done or trunc: break
        heatmap.add_episode(epi)


        print("Epsiode total reward: ", reward)
        wins += (info['winner'] == 1)
        algo.on_end_game(result=None, stats=None)
        agent2.on_end_game(result=None, stats=None)        

    print(f'SAVING AT {opts.heatmap_path}')
    heatmap.save(opts.heatmap_path, opts.title, opts.suffix)


if __name__ == '__main__':
    generate()
