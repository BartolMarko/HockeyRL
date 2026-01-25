import torch
import numpy as np
import time
import random
from pathlib import Path
from omegaconf import OmegaConf

from src.agent_factory import agent_factory
from src.environments import environment_factory
from src.evaluation import Evaluator
from src.opponent_pool import opponent_pool_factory
from src.training_monitor import TrainingMonitor

from src.TDMPC.tdmpc import TDMPC
from src.TDMPC.helper import ReplayBuffer
from src.TDMPC.agent import TDMPCAgent

torch.backends.cudnn.benchmark = True

# TODO: MAJOR: add copy of TDMPC1 code for backward compatibility
CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "tdmpc2.yaml"


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

    env = environment_factory(cfg.env_name)
    cfg = add_env_variables_to_config(env, cfg)

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
    # TODO: Use different copies to parallelize evaluation and training?

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
    mirror_episodes = cfg.get("mirror_episodes", False)
    episode_idx, step = 0, 0
    last_update_step, last_eval_step, last_save_step, last_selfplay_step = 0, 0, 0, 0
    while step < cfg.train_steps:
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
            cfg.get("selfplay", False)
            and step >= cfg.selfplay_start_step
            and (step - last_selfplay_step) >= cfg.selfplay_freq
        ):
            opponent_pool.add_opponent(
                get_copy_of_self(tdmpc, step, cfg),
                removable=cfg.get("self_removable", True),
            )
            evaluation_opponents.append(get_copy_of_self(tdmpc, step, cfg))
            last_selfplay_step = step
            print(f"Added self-play agent to opponent pool at step {step}.")

        episode_start_time = time.time()

        agent = TDMPCAgent(
            load_dir=None,
            tdmpc=tdmpc,
            step=step,
            eval_mode=False,
            name_suffix=f"_step_{step}",
        )
        opponent = opponent_pool.sample_opponent()
        episode, _ = evaluator.run_episode(
            env,
            agent,
            opponent,
            render_mode=None,
        )
        buffer += episode

        opponent_pool.add_episode_outcome(opponent, episode.outcome)
        training_monitor.log_training_episode(opponent.name, episode, step, episode_idx)
        step += len(episode)
        episode_idx += 1

        if mirror_episodes:
            episode.mirror()
            buffer += episode
            step += len(episode)
            episode_idx += 1
            # TODO: log mirrored episode?
            # TODO: add mirrored episode outcome to opponent pool?
            # So far I don't do it, it would lead to doubling statistics
            # This way, stuff is logged only once every 2 episodes

        env_step = int(step * cfg.action_repeat)
        # TODO: WATCH OUT FOR ENV STEP WHEN USING ACTION REPEAT

        # Update model
        # TODO: Average train metrics, not overwrite
        train_metrics = {}
        if step >= cfg.seed_steps:
            num_updates = step - last_update_step
            last_update_step = step
            for i in range(num_updates):
                train_metrics.update(tdmpc.update(buffer, step + i))

        # Log training metrics
        train_metrics = {f"Losses/{k}": v for k, v in train_metrics.items()}
        training_monitor.run.log(train_metrics, step=step)

        print(
            f"Step {step}.",
            f"Episode {episode_idx} finished in {time.time() - episode_start_time:.2f}s.",
            f"Opponent: {opponent.name}.",
            f"Episode outcome: {episode.outcome.name}.",
            sep=" ",
        )

        # Evaluate agent periodically
        # TODO: Measure eval time in Evaluator and log it
        if env_step - last_eval_step >= cfg.eval_freq or step >= cfg.train_steps:
            eval_start_time = time.time()
            final_evaluation = step >= cfg.train_steps
            save_episodes_per_outcome = (
                cfg.video_episodes_per_outcome if final_evaluation else 0
            )

            agent.eval_mode = True
            evaluator.evaluate_agent_and_save_metrics(
                env,
                agent,
                evaluation_opponents,
                num_episodes=cfg.eval_episodes_per_opponent,
                render_mode="rgb_array" if final_evaluation else None,
                save_heatmaps=cfg.get("save_heatmaps", False) or final_evaluation,
                wandb_run=training_monitor.run,
                train_step=env_step,
                save_episodes_per_outcome=save_episodes_per_outcome,
            )
            last_eval_step = env_step
            print(
                f"Evaluation at step {step} (env step {env_step}) completed in {time.time() - eval_start_time:.2f}s."
            )

        if cfg.save_model and (
            (env_step - last_save_step >= cfg.save_model_freq)
            or step >= cfg.train_steps
        ):
            tdmpc.save_to_wandb(training_monitor.run, step=step)
            last_save_step = env_step

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
