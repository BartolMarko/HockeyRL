import torch
import numpy as np
import time
import random
from pathlib import Path
from omegaconf import OmegaConf

from src.environments import SparseRewardHockeyEnv
from src.evaluation import Evaluator
from src.named_agent import WeakBot, StrongBot
from src.opponent_pool import OpponentPoolThompsonSampling
from src.training_monitor import TrainingMonitor

from src.TDMPC.tdmpc import TDMPC
from src.TDMPC.helper import ReplayBuffer
from src.TDMPC.agent import TDMPCAgent

torch.backends.cudnn.benchmark = True

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default.yaml"
RUN_NAME = "tdmpc_baseline_weak_and_strong_bots_only"
MODEL_CHECKPOINTS_DIR = Path("./model_checkpoints")


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
    return cfg


def train(cfg):
    """Training script for TD-MPC"""
    # TODO: Add choice of opponents and env to config
    # TOOD: Add run_name to config

    assert torch.cuda.is_available()
    set_seed(cfg.seed)

    env = SparseRewardHockeyEnv()
    cfg = add_env_variables_to_config(env, cfg)

    tdmpc, buffer = TDMPC(cfg), ReplayBuffer(cfg)
    evaluator = Evaluator(device=cfg.device)

    weak_bot = WeakBot()
    strong_bot = StrongBot()
    opponent_pool = OpponentPoolThompsonSampling(
        opponents=[weak_bot, strong_bot],
        window_size_episodes=cfg.opponent_pool_window_size,
    )
    training_monitor = TrainingMonitor(
        run_name=RUN_NAME,
        config=OmegaConf.to_container(cfg),
    )

    # Run training
    episode_idx, step = 0, 0
    last_update_step, last_eval_step, last_save_step = 0, 0, 0
    while step < cfg.train_steps:
        episode_start_time = time.time()

        agent = TDMPCAgent(
            load_dir=None,
            tdmpc=tdmpc,
            step=step,
            eval_mode=False,
            name_suffix=f"_step_{step}",
        )
        opponent = opponent_pool.sample_opponent(
            draw_weight=cfg.opponent_pool_draw_weight
        )
        episode, _ = evaluator.run_episode(
            env,
            agent,
            opponent,
            render_mode=None,
        )
        buffer += episode

        opponent_pool.add_episode_outcome(opponent, episode.outcome)
        training_monitor.log_training_episode(opponent.name, episode, step, episode_idx)

        # Update model
        # TODO: Average train metrics, not overwrite
        train_metrics = {}
        if step >= cfg.seed_steps:
            num_updates = step - last_update_step
            last_update_step = step
            for i in range(num_updates):
                train_metrics.update(tdmpc.update(buffer, step + i))

        step += len(episode)
        episode_idx += 1
        env_step = int(step * cfg.action_repeat)
        # TODO: WATCH OUT FOR ENV STEP WHEN USING ACTION REPEAT

        # Log training metrics
        train_metrics = {f"Losses/{k}": v for k, v in train_metrics.items()}
        training_monitor.run.log(train_metrics, step=step)

        print(
            f"Step {step}. Episode {episode_idx} finished in {time.time() - episode_start_time:.2f}s."
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
                opponent_pool.get_opponents(),
                num_episodes=cfg.eval_episodes_per_opponent,
                render_mode="rgb_array" if final_evaluation else None,
                save_heatmaps=final_evaluation,
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
            MODEL_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
            checkpoint_path = MODEL_CHECKPOINTS_DIR / f"tdmpc_step_{step}.pt"
            tdmpc.save(checkpoint_path)
            last_save_step = env_step
            print(f"Saved model checkpoint to {checkpoint_path}.")
    training_monitor.finish_training()

    print("Training completed successfully")


if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as f:
        cfg = OmegaConf.load(f)
        train(cfg)
