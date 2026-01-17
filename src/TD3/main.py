import argparse, os
import pickle
import random
from pathlib import Path

import torch, numpy as np
from gymnasium import spaces

from hockey.hockey_env import HockeyEnv

from src.TD3.td3 import TD3
from src.TD3.schedule import SchedulerFactory
from src.TD3.opponent_scheduler import OpponentSchedulerFactory
from src.TD3.noise import NosieFactory
from src.TD3.config_reader import Config

from src.episode import Episode
from src.training_monitor import TrainingMonitor
from src.named_agent import WeakBot
from src.evaluation import Evaluator


RUN_NAME = "td3_self_play"

def set_seed(random_seed):
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.cuda.manual_seed(random_seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', action='store', type=str,
                        default='config.yaml', help='Config file')

    args= parser.parse_args()
    cfg = Config(args.config)

    t_cfg = cfg.training
    env = HockeyEnv()
    env_name = env.__class__.__name__

    log_interval = 20
    total_timesteps = t_cfg['total_timesteps']
    max_episode_length = t_cfg['max_ep_length']

    set_seed(t_cfg.get('seed'))

    action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    noise_scheduler = SchedulerFactory.get_scheduler(t_cfg['noise_scheduler']) 

    noise_sampler = NosieFactory.get_noise(t_cfg['action_noise'], action_dim=action_space.shape[0])

    opp_scheduler = OpponentSchedulerFactory.get_scheduler(cfg, env.observation_space, action_space,
                                                            on_phase_change=noise_scheduler.reset)

    td3 = TD3(env.observation_space, action_space, cfg['td3'])

    print(next(td3.model.actor.parameters()).device)
    
    def get_action(t, observation):
        action = td3.act(observation)
        if noise_scheduler is not None:
            noise_scale = noise_scheduler.value()
            action += noise_sampler.sample() * noise_scale
        return np.clip(action, action_space.low, action_space.high)

    
    if t_cfg.get('resume_from') is not None:
        print("Loading checkpoint: ", t_cfg['resume_from']['checkpoint'])
        td3.restore_state( torch.load(t_cfg['resume_from']['checkpoint']))

    k = 0

    epi_rewards = np.zeros(log_interval)
    win_info = np.zeros(log_interval)
    last_n_winrates = np.zeros(log_interval)
    
    os.makedirs(t_cfg.result_directory, exist_ok=True)

    if t_cfg.get('resume_from') is not None:
        i_episode = t_cfg.resume_from.start_epi
        start_idx = t_cfg.resume_from.start_timestep
    else:
        i_episode = 0
        start_idx = 0

    ob, _info = env.reset()
    ob_agent2 = env.obs_agent_two()
    noise_sampler.reset()
    total_reward, total_length=(0,)*2

    if t_cfg.use_opp_scheduler:
        print('start_idx', start_idx)
        agent2 = opp_scheduler.get_opponent(start_idx)
    else:
        agent2 = WeakBot()

    episode = Episode(ob)
    training_monitor = TrainingMonitor(
        run_name=RUN_NAME,
        config=cfg
    )
    evaluator = Evaluator(torch.device('cuda'))
    for t in range(1 + start_idx, total_timesteps + 1):
        if t > t_cfg.start_after:
            a1 = get_action(t, ob)
        else:
            a1 = action_space.sample()
        
        a2 = agent2.act(ob_agent2)
        (ob_new, reward, done, trunc, info) = env.step(np.hstack([a1, a2]))
        total_reward+= reward
        td3.store_transition((ob, a1, reward, ob_new, done))
        ob=ob_new
        ob_agent2 = env.obs_agent_two()
        episode.add(ob, a1, a2, reward, done)
        if done or trunc or total_length == max_episode_length: 
            ob, _ = env.reset()
            ob_agent2 = env.obs_agent_two()
            i_episode += 1
            win_info[(i_episode - 1) % log_interval] = (info['winner'])# == 1)
            epi_rewards[(i_episode - 1) % log_interval] = total_reward
            total_reward, total_length = (0,)*2

            if i_episode % 1000 == 0:
                print("########## Saving a checkpoint... ##########")
                td3.save_to_wandb(training_monitor.run, step=i_episode)

            training_monitor.log_training_episode(opponent_name=agent2.name,
                                                  episode=episode,
                                                  step=t,
                                                  episode_index=i_episode)

            if i_episode % log_interval == 0:
                winrate  = (win_info == 1).mean()
                last_n_winrates[k % log_interval] = winrate
                k += 1


                if last_n_winrates.mean() > .88:
                    if hasattr(opp_scheduler, "trigger_phase_change"):
                        if opp_scheduler.trigger_phase_change():
                            torch.save(td3.state(), f'./{t_cfg.result_directory}/td3_{env_name}_{i_episode}-s{random_seed}.pth')
                            print(f"triggered phase change at episode: {i_episode}")
            
            if t_cfg.use_opp_scheduler:
                agent2 = opp_scheduler.get_opponent(t)

            noise_sampler.reset()
            episode = Episode(ob)

        if t % t_cfg.opp_update_freq == 0:
            opp_scheduler.add_agent(td3.get_policy_config(), f"self_play_t_{t}")

        if t > t_cfg.start_after and t % t_cfg.update_every == 0:
            q_losses = []
            pi_losses = []
            for j in range(t_cfg.update_every):
                loss_q, loss_pi = td3.update(j)
                q_losses.append(loss_q)
                if loss_pi is not None:
                    pi_losses.append(loss_pi)

            mean_loss_q  = np.array(q_losses).mean()
            mean_loss_pi = np.array(pi_losses).mean()

            training_monitor.run.log(
                {"loss_q": mean_loss_q,
                "loss_pi": mean_loss_pi},
                step = t
            )


        if t % t_cfg['eval_freq'] == 0:
            final_eval = (t == total_timesteps)
            save_epi_per_outcome = (t_cfg['video_episodes_per_outcome'] if 
                                    final_eval else 0)
            evaluator.evaluate_agent_and_save_metrics(
                env,
                td3,
                opp_scheduler.get_opponents(),
                num_episodes=t_cfg.eval_episodes_per_opp,
                render_mode = None,
                save_heatmaps= final_eval,
                wandb_run=training_monitor.run,
                train_step=t,
                save_episodes_per_outcome=save_epi_per_outcome
            )


if __name__ == '__main__':
    main()