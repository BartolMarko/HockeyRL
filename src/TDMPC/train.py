import torch
import numpy as np
import time
import random
from pathlib import Path
from omegaconf import OmegaConf
from hockey.hockey_env import HockeyEnv

from src.agent_factory import agent_factory
from src.environments import environment_factory, env_reward_wrapper
from src.evaluation import Evaluator
from src.opponent_pool import opponent_pool_factory
from src.training_monitor import TrainingMonitor

from src.TDMPC.tdmpc import TDMPC
from src.TDMPC.helper import ReplayBuffer
from src.TDMPC.agent import TDMPCAgent

torch.backends.cudnn.benchmark = True

# TODO: MAJOR: add copy of TDMPC1 code for backward compatibility
CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "tdmpc2_defense.yaml"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_env_variables_to_config(env, cfg):
    """Add environment-specific variables to the config."""
    cfg.obs_dim = env.observation_space.shape[0]
    cfg.obs_shape = env.observation_space.shape

    cfg.action_dim = env.action_space.shape[0] // 2
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    return cfg


def get_copy_of_self(
    tdmpc: TDMPC,
    step: int,
    cfg: OmegaConf,
):
    """Add the current TD-MPC agent to the opponent pool for self-play."""
    tdmpc_copy = TDMPC(cfg)
    tdmpc_copy.load_state_dict(tdmpc.state_dict())

    self_opponent = TDMPCAgent(
        load_dir=None,
        tdmpc=tdmpc_copy,
        step=step,
        eval_mode=True,
        name_suffix=f"_selfplay_step_{step}",
    )
    return self_opponent


def train(cfg):
    """Training script for TD-MPC"""
    assert torch.cuda.is_available()
    set_seed(cfg.seed)

    evaluation_env = HockeyEnv()
    cfg = add_env_variables_to_config(evaluation_env, cfg)
    train_envs = []
    for env_name in cfg.train_env_list:
        current_env = environment_factory(env_name)
        train_envs.append(
            env_reward_wrapper(
                current_env, cfg.reward_name, **cfg.get("reward_kwargs", {})
            )
        )

    training_opponents_configs = {
        opp_name: opp_cfg
        for opp_name, opp_cfg in cfg.opponents.items()
        if opp_cfg.get("train_start_step", None) is not None
    }
    training_opponents_configs = sorted(
        training_opponents_configs.items(),
        key=lambda item: item[1]["train_start_step"],
    )  # List of (name, config) tuples

    evaluation_opponent_configs = {
        opp_name: opp_cfg
        for opp_name, opp_cfg in cfg.opponents.items()
        if opp_cfg.get("evaluate_against", False)
    }
    evaluation_opponents = [
        agent_factory(opp_name, opp_cfg)
        for opp_name, opp_cfg in evaluation_opponent_configs.items()
    ]
    # Different copies of opponents for evaluation and training

    opponent_pool = opponent_pool_factory(
        cfg.get("opponent_pool", "ThompsonSampling"),
        opponents=[],
        window_size_episodes=cfg.get("opponent_pool_window_size", 100),
        draw_weight=cfg.get("opponent_pool_draw_weight", 0.5),
        max_num_opponents=cfg.get("max_num_opponents_in_pool", 10),
    )

    tdmpc, buffer = TDMPC(cfg), ReplayBuffer(cfg)
    evaluator = Evaluator(device=cfg.device)
    training_monitor = TrainingMonitor(
        run_name=cfg.run_name,
        config=OmegaConf.to_container(cfg),
    )

    # Run training
    action_repeat = cfg.get("action_repeat", 1)
    selfplay = cfg.get("selfplay", False)
    selfplay_start_step = cfg.get("selfplay_start_step", 0)
    evaluate_against_self = cfg.get("evaluate_against_self", False)
    mirror_episodes = cfg.get("mirror_episodes", False)
    soft_winrate_threshold = cfg.get("add_self_winrate_soft_threshold", 200.0)
    # If no threshold given, only add self at fixed intervals

    episode_idx, step = 0, 0
    last_update_step, last_eval_step, last_save_step, last_selfplay_step = 0, 0, 0, 0
    while step < cfg.train_steps:
        current_winrate_against_pool = (
            training_monitor.get_lowest_winrate_against_opponents(
                opponent_pool.get_opponent_names()
            )
        )
        while (
            len(training_opponents_configs)
            and step >= training_opponents_configs[0][1]["train_start_step"]
        ):
            opponent_name, opponent_cfg = training_opponents_configs.pop(0)
            opponent_pool.add_opponent(
                agent_factory(opponent_name, opponent_cfg),
                removable=opponent_cfg.get("removable", False),
            )
            print(
                f"Added new opponent '{opponent_name}' to opponent pool at step {step}."
            )

        if (
            selfplay
            and step >= selfplay_start_step
            and (
                (step - last_selfplay_step) >= cfg.selfplay_freq
                or current_winrate_against_pool >= soft_winrate_threshold
            )
        ):
            opponent_pool.add_opponent(
                get_copy_of_self(tdmpc, step, cfg),
                removable=cfg.get("self_removable", True),
            )
            if evaluate_against_self:
                evaluation_opponents.append(get_copy_of_self(tdmpc, step, cfg))
            last_selfplay_step = step
            print(f"Added self-play agent to opponent pool at step {step}.")

        train_episodes_playing_start_time = time.time()
        agent = TDMPCAgent(
            load_dir=None,
            tdmpc=tdmpc,
            step=step,
            eval_mode=False,
            name_suffix=f"_step_{step}",
        )
        opponent = opponent_pool.sample_opponent()

        steps_played = 0
        for env in train_envs:
            episode, _ = evaluator.run_episode(
                env,
                agent,
                opponent,
                render_mode=None,
                every_n_steps=action_repeat,
            )
            buffer += episode

            opponent_pool.add_episode_outcome(opponent, episode.outcome)
            training_monitor.log_training_episode(
                opponent.name, episode, step, episode_idx
            )
            step += len(episode)
            steps_played += len(episode)
            episode_idx += 1

            print(
                f"Step {step}.",
                f"Episode {episode_idx}.",
                f"Opponent: {opponent.name}.",
                f"Episode outcome: {episode.outcome.name}.",
                sep=" ",
            )

            if mirror_episodes:
                episode.mirror()
                buffer += episode
                step += len(episode)
                episode_idx += 1

        train_episodes_playing_duration = (
            time.time() - train_episodes_playing_start_time
        )

        # Update model
        update_start_time = time.time()
        train_metrics = {}
        num_updates = 0
        if step >= cfg.seed_steps:
            num_updates = step - last_update_step
            last_update_step = step
            for i in range(num_updates):
                train_metrics.update(tdmpc.update(buffer, last_update_step + i))
        update_duration = time.time() - update_start_time

        # Log training metrics
        time_metrics = {
            "Time/train_episodes_playing": train_episodes_playing_duration,
            "Time/train_episodes_per_step": train_episodes_playing_duration
            / max(steps_played, 1),
            "Time/model_update": update_duration,
            "Time/model_updates_per_episode_step": update_duration
            / max(num_updates, 1),
        }
        train_metrics = {f"Losses/{k}": v for k, v in train_metrics.items()}
        training_monitor.run.log({**train_metrics, **time_metrics}, step=step)

        if cfg.save_model and (
            (step - last_save_step >= cfg.save_model_freq) or step >= cfg.train_steps
        ):
            tdmpc.save_to_wandb(training_monitor.run, step=step)
            last_save_step = step

        # Evaluate agent periodically
        if step - last_eval_step >= cfg.eval_freq or step >= cfg.train_steps:
            eval_start_time = time.time()
            final_evaluation = step >= cfg.train_steps
            save_episodes_per_outcome = (
                cfg.video_episodes_per_outcome if final_evaluation else 0
            )

            agent.eval_mode = True
            evaluator.evaluate_agent_and_save_metrics(
                evaluation_env,
                agent,
                evaluation_opponents,
                num_episodes=cfg.eval_episodes_per_opponent,
                render_mode="rgb_array" if final_evaluation else None,
                save_heatmaps=final_evaluation,
                wandb_run=training_monitor.run,
                train_step=step,
                save_episodes_per_outcome=save_episodes_per_outcome,
            )
            last_eval_step = step
            print(
                f"Evaluation at step {step} completed in {time.time() - eval_start_time:.2f}s."
            )

    training_monitor.finish_training()
    print("Training completed successfully")


if __name__ == "__main__":
    major_cc, minor_cc = torch.cuda.get_device_capability()
    if major_cc >= 8:
        torch.set_float32_matmul_precision("high")
        print(f"Set float32 matmul precision to high. CUDA: {major_cc}.{minor_cc}")

    with open(CONFIG_PATH, "r") as f:
        cfg = OmegaConf.load(f)
        train(cfg)
