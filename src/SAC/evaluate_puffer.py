import os
import sys
sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
from omegaconf import OmegaConf
import time
from hockey import hockey_env as h_env
import imageio
from . import helper
from . import puffer_wrapper as pfw


def evaluate(cfg, agent, opponent, vec_env, logger=None, episode_index=None, episodes_per_env=None):
    start_time = time.time()
    agent.eval()

    if episodes_per_env is None:
        episodes_per_env = 1
    num_episodes = vec_env.num_envs * episodes_per_env
    assert num_episodes > 0, \
        "Number of episodes must be positive. episodes_per_env = {}".format(
                episodes_per_env)

    win_counts = np.zeros(vec_env.num_envs, dtype=int)
    lose_counts = np.zeros(vec_env.num_envs, dtype=int)
    draw_counts = np.zeros(vec_env.num_envs, dtype=int)
    episodes_done = np.zeros(vec_env.num_envs, dtype=int)
    obs_batch, _ = vec_env.reset()
    obs_opponent_batch = vec_env.obs_agent_two()
    done_batch = np.zeros(vec_env.num_envs, dtype=bool)
    active_envs = np.ones(vec_env.num_envs, dtype=bool)
    done_count = 0

    episode_scores = 0.0
    episode_lengths = 0

    while done_count < num_episodes:
        agent_actions = agent.plan_batch(obs_batch, eval_mode=True,
                                         step=episode_index)
        if logger is not None:
            logger.add_state(obs_batch)
            logger.add_action(agent_actions)

        opponent_actions = opponent.plan_batch(obs_opponent_batch)
        combined_actions = np.hstack([agent_actions, opponent_actions])
        combined_actions = np.clip(combined_actions, -1, 1).astype(np.float32)

        (obs_batch, rewards_batch, done_batch,
         truncated_batch, info_batch) = vec_env.step(combined_actions)

        active_envs = (episodes_done < episodes_per_env)
        # We should count result if it is terminated OR truncated
        real_done_batch = np.logical_or(done_batch, truncated_batch)
        finished_batch = np.logical_and(real_done_batch, active_envs)
        if 'final_info' in info_batch:
            for idx, infos in enumerate(info_batch['final_info']):
                if infos is None:
                    continue
                winner = infos.get('winner')
                if winner == 1:
                    win_counts[idx] += 1
                elif winner == -1:
                    lose_counts[idx] += 1
                elif winner == 0:
                    draw_counts[idx] += 1
        else:
            # unlikely, but lets keep this
            win_counts += (finished_batch & (rewards_batch > 0)).astype(int)
            lose_counts += (finished_batch & (rewards_batch < 0)).astype(int)
            draw_counts += (finished_batch & (rewards_batch == 0)).astype(int)

        episodes_done += finished_batch.astype(int)
        done_count += np.sum(finished_batch)

        if isinstance(vec_env, pfw.HockeyVecEnv):
            # Ensure we are using the wrapper
            obs_opponent_batch = vec_env.obs_agent_two()
        else:  # Fallback
            obs_opponent_batch = info_batch.get('obs_agent_two')

        episode_scores += rewards_batch.sum()
        episode_lengths += np.sum(active_envs)

    if hasattr(opponent, 'record_play_scores'):
        total_wins = int(np.sum(win_counts))
        total_losses = int(np.sum(lose_counts))
        total_draws = int(np.sum(draw_counts))
        opponent.record_play_scores(total_losses, total_wins, total_draws)
    total_episodes = np.sum(win_counts) + np.sum(lose_counts) + np.sum(draw_counts)
    episode_metrics = {}
    episode_metrics['episode_score'] = episode_scores / total_episodes
    episode_metrics['episode_length'] = episode_lengths / total_episodes
    episode_metrics['win'] = np.sum(win_counts)
    episode_metrics['lose'] = np.sum(lose_counts)
    episode_metrics['draw'] = np.sum(draw_counts)
    episode_metrics['total_episodes'] = total_episodes
    end_time = time.time()
    episode_metrics['episode_time'] = (end_time - start_time) / total_episodes
    return episode_metrics

def puffer_evaluate_against_pool(env, agent, opponent_pool,
                                 num_episodes: int = 100,
                                 step: int | None = None, logger=None):
    overall_stats = {}
    players = opponent_pool.get_playable_opponents()
    print("[EVAL] Starting evaluation against opponent pool...:", [
        p.name for p in players])
    for opponent in players:
        if opponent.is_mgr():
            pool = opponent.get_last_n_opponents(20)
            results = puffer_evaluate_against_pool(
                    env, agent, pool,
                    num_episodes=num_episodes,
                    logger=logger,
                    step=step)
            win_count = sum([results[op]['win'] for op in results])
            lose_count = sum([results[op]['lose'] for op in results])
            draw_count = sum([results[op]['draw'] for op in results])
            overall_stats[opponent.name] = {
                'win': win_count,
                'lose': lose_count,
                'draw': draw_count
            }
        else:
            episodes_per_env = max(1, num_episodes // env.num_envs)
            metrics = evaluate(agent.cfg, agent, opponent, env,
                               episode_index=step, logger=logger,
                               episodes_per_env=episodes_per_env)
            overall_stats[opponent.name] = {
                'win': metrics['win'],
                'lose': metrics['lose'],
                'draw': metrics['draw']
            }
            if opponent.pool:
                opponent.pool.record_play_scores(metrics['lose'],
                                                 metrics['win'],
                                                 metrics['draw'],
                                                 episode_index=opponent.ep)
        print(f"[EVAL] vs {opponent.name}: Wins: {overall_stats[opponent.name]['win']}, Losses: {overall_stats[opponent.name]['lose']}, Draws: {overall_stats[opponent.name]['draw']}.")
    return overall_stats

def view_gameplay(env, agent, opponent, render: bool = True, video_path: str|None = None, num_episodes: int = 5):
    wins, lose, draw = 0, 0, 0
    frames = []
    render_mode = 'human' if render else 'rgb_array'
    env = h_env.HockeyEnv()
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        obs_opponent = env.obs_agent_two()
        while not done:
            if render or video_path is not None:
                frames.append(env.render(mode=render_mode))
            if render:
                time.sleep(0.02)
            agent_action = agent.act(obs)
            opponent_action = opponent.act(obs_opponent)
            combined_action = np.hstack([agent_action, opponent_action])
            combined_action = np.clip(combined_action, -1, 1).astype(np.float32)
            obs, reward, done, truncated, info = env.step(combined_action)
            obs_opponent = env.obs_agent_two()
        winner = info.get('winner')
        if winner == 1:
            wins += 1
        elif winner == -1:
            lose += 1
        elif winner == 0:
            draw += 1
    if video_path is not None:
        imageio.mimwrite(video_path, frames, fps=30, format='gif')
    return wins, lose, draw

def save_gameplay_video(env, agent, opponent, video_path, num_episodes: int = 1):
    return view_gameplay(env, agent, opponent, render=False, video_path=video_path, num_episodes=num_episodes)

def main(args):
    if len(args) < 2:
        print("Usage: python evaluate_puffer.py <left_agent_name> <right_agent_name>")
        print("Usage: python evaluate_puffer.py -e <left_agent_name> <right_agent_name>")
        sys.exit(1)
    render = True
    num_episodes = 10
    save_path = None
    if args[0] == '-e':
        render = False
        num_episodes = 100
        args = args[1:]
    elif args[0] == '-s':
        render = False
        num_episodes = 1
        save_path = 'gameplay__{}vs{}.gif'
        args = args[1:]
    left_agent = args[0]
    right_agent = args[1]

    env = h_env.HockeyEnv()
    def get_agent_by_name(name):
        if name.lower() == 'strongbot':
            return h_env.BasicOpponent(weak=False)
        elif name.lower() == 'weakbot':
            return h_env.BasicOpponent(weak=True)
        elif name.lower() == 'puckfollowbot':
            from adversarial import create_puck_follow_bot
            return create_puck_follow_bot()
        elif name.startswith('sac'):
            return helper.load_agent_from_config(name, env)
        elif name == "oldsac":
            from KaranhanS.load_model import get_agent
            return get_agent(env)
        elif name.startswith('td3'):
            from src.agent_factory import agent_factory
            cfg_path = name
            if not name.endswith('.yaml'):
                cfg_path = f"{name}.yaml"
            cfg = OmegaConf.load(cfg_path)
            return agent_factory(name, cfg)
        else:
            raise ValueError(f"Unknown agent name: {name}")

    agent = get_agent_by_name(left_agent)

    opponent_list = []
    if right_agent.endswith('.yaml'):
        opp_cfg = OmegaConf.load(right_agent)
        opponents = opp_cfg.opponents
        for opponent_cfg in opponents:
            o_type = opponent_cfg.type
            if o_type in ['WeakBot', 'StrongBot', 'PuckFollowBot']:
                opponent = get_agent_by_name(o_type)
                opponent.name = o_type
            elif o_type == 'CustomAgent':
                name = opponent_cfg.experiment_name
                opponent = get_agent_by_name(name)
            elif o_type == 'TD3':
                from src.agent_factory import agent_factory
                cfg_path = opponent_cfg.config_path
                cfg = OmegaConf.load(cfg_path)
                opponent = agent_factory(cfg_path.replace('.yaml', ''), cfg)
            else:
                opponent = get_agent_by_name(o_type)
            assert opponent is not None, \
                f"Failed to create opponent for config: {opponent_cfg}"
            opponent_list.append(opponent)
    else:
        opponent = get_agent_by_name(right_agent)
        opponent_list.append(opponent)
    env.close()

    total_wins = 0
    total_losses = 0
    total_draws = 0
    total_games = 0
    for opponent in opponent_list:
        if save_path is not None:
            d_save_path = save_path.format(agent.name, opponent.name)
        else:
            d_save_path = None
        win, lose, draw = view_gameplay(env, agent, opponent, render=render,
                                        video_path=d_save_path,
                                        num_episodes=num_episodes)
        print(f"L: {agent.name} wins {win}")
        print(f"R: {opponent.name} wins {lose}")
        print(f"Draws = {draw}")
        print(f"Total = {win + lose + draw}")
        total_wins += win
        total_losses += lose
        total_draws += draw
        total_games += win + lose + draw

    performance = (total_wins - total_losses) / total_games
    print(f"Overall: {agent.name} wins {total_wins}, loses {total_losses}"
          f", draws {total_draws}, total {total_games}, performance {performance:.2f}")


if __name__ == '__main__':
    import sys
    arguments = sys.argv[1:]
    main(arguments)
