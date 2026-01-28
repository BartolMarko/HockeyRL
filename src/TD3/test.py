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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    parser.add_argument('-g', '--gif', action='store_true',
                        help='render env into gif')
    parser.add_argument('--output', action='store',
                        default='test_output')

    return parser.parse_args()



def test():

    opts = parse_args()

    cfg = Config(opts.config)['td3']
    opp_cfg = Config(opts.opp_config)['td3']
    print(opp_cfg)

    save_gif = opts.gif

    env = h_env.HockeyEnv()
    out_dir = opts.output

    os.makedirs(out_dir, exist_ok=True)

    if save_gif:
        render_mode = 'rgb_array'
        gif_dir = os.path.join(out_dir, 'gif')
        os.makedirs(gif_dir, exist_ok=True)

    opp_type = opts.opponent.lower()
    
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    cfg['observation_space'] = env.observation_space
    cfg['action_space'] = action_space
    opp_cfg['observation_space'] = env.observation_space
    opp_cfg['action_space'] = action_space
    

    match opp_type:
        case 'weak':
            agent2 = BasicOpponent(weak=True)
        case 'strong':
            agent2 = BasicOpponent(weak=False)
        case 'custom':
            agent2 = CustomOpponent()
        case _:
            if os.path.exists(opp_type):
                agent2 = TD3(opp_cfg)
                print(agent2.model.actor)
                print("loading agent2")
                agent2.restore_state(torch.load(opp_type))
                opp_type = 'td3' # override for gif path
            else:
                raise Exception('Invalid opponent '+ opts.opponent)

    # agent2 = BasicOpponent(weak=weak)

    if opts.path == 'custom':
        algo = CustomOpponent()
    else:
        algo = TD3(cfg)
        algo.restore_state(torch.load(opts.path))

    wins = 0
    max_epi = opts.maxepisodes
    max_timesteps = opts.maxtimesteps

    for i_epi in range(1, max_epi + 1):
        ob, _info = env.reset()
        ob_agent2 = env.obs_agent_two()
        ep_reward = 0
        images = []

        for t in range(max_timesteps):
            a1 = algo.act(ob)
            a2 = agent2.act(ob_agent2)

            (ob_new, reward, done, trunc, info) = env.step(np.hstack([a1, a2]))

            ob = ob_new
            ob_agent2 = env.obs_agent_two()

            ep_reward += reward

            if save_gif:
                img = env.render(render_mode)
                img = Image.fromarray(img)
                images.append(img)

            if done or trunc: break

        print("Epsiode total reward: ", reward)
        wins += (info['winner'] == 1)

        if images:
            images[0].save(
                f'./{gif_dir}/{i_epi:02}-{t:03}-{opp_type}.gif',
                save_all=True,
                append_images=images[1:],
                duration=33,
                loop=0 
            )
    print("winrate: ", wins/max_epi)
        


if __name__ == '__main__':
    test()
