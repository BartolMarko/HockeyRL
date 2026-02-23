import torch
import numpy as np
import time
import random
import os
from pathlib import Path
from omegaconf import OmegaConf
from hockey.hockey_env import HockeyEnv
import wandb

from src.named_agent import NamedAgent
from src.agent_factory import agent_factory, create_ppo_ensemble_agent
from src.environments import (
    environment_factory,
    env_reward_wrapper,
    SparseRewardHockeyEnv,
)
from src.evaluation import Evaluator
from src.opponent_pool import opponent_pool_factory
from src.training_monitor import TrainingMonitor
from src.ensembles.ppo_ensemble_agent import PPOEnsembleAgent
from src.episode import Episode

torch.backends.cudnn.benchmark = True

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "ppo_ensemble_config.yaml"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_copy_of_self(wandb_run: wandb.Run, agent: PPOEnsembleAgent, step: int):
    opponent_folder_name = f"self_opponent_step_{step}"
    agent.save_to_wandb(wandb_run, folder_name=opponent_folder_name)
    return create_ppo_ensemble_agent(
        name=f"self_PPOEnsemble_step_{step}",
        cfg=None,
        load_dir=os.path.join(wandb_run.dir, opponent_folder_name),
    )


def run_episode(
    env: HockeyEnv,
    agent: PPOEnsembleAgent,
    opponent: NamedAgent,
    agent_repeat: int,
    ppo_actions_counter: dict[str, int],
):
    obs, _ = env.reset()
    obs_opponent = env.obs_agent_two()
    episode = Episode(obs)

    agent.ppo.episode_start(obs)
    opponent.on_start_game(episode.id)

    while not episode.done:
        agent_actions = agent.get_agent_actions(obs)
        extended_obs = agent.extend_state_with_q_values(obs, agent_actions)
        with torch.no_grad():
            ppo_action, logprob, _, value = agent.ppo.actor_critic.get_action_and_value(
                torch.Tensor(extended_obs).to(agent.ppo.device)
            )
        ppo_action = ppo_action.cpu().numpy().item()
        logprob = logprob.cpu().numpy().item()
        value = value.cpu().numpy().item()

        for _ in range(agent_repeat):
            action = agent.agents[ppo_action].get_step(obs)
            ppo_actions_counter[agent.agents[ppo_action].name] += 1
            opponent_action = opponent.get_step(obs_opponent)
            obs, reward, done, _, info = env.step(np.hstack([action, opponent_action]))
            obs_opponent = env.obs_agent_two()

            episode.add(obs, action, opponent_action, reward, done)
            if done:
                break

        agent.ppo.add_to_storage(
            obs=extended_obs, action=ppo_action, logprob=logprob, value=value, done=done
        )

    opponent.on_end_game(result=None, stats=None)
    return episode


def train(cfg):
    """Training script for TD-MPC"""
    assert torch.cuda.is_available()
    set_seed(cfg.seed)

    agent = create_ppo_ensemble_agent(name="PPOEnsemble", cfg=cfg)
    evaluation_env = SparseRewardHockeyEnv()
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

    evaluator = Evaluator(device=cfg.device)
    training_monitor = TrainingMonitor(
        run_name=cfg.run_name,
        config=OmegaConf.to_container(cfg),
    )

    # Run training
    selfplay = cfg.get("selfplay", False)
    selfplay_start_step = cfg.get("selfplay_start_step", 0)
    evaluate_against_self = cfg.get("evaluate_against_self", False)
    soft_winrate_threshold = cfg.get("add_self_winrate_soft_threshold", 200.0)
    # If no threshold given, only add self at fixed intervals

    episode_idx, step = 0, 0
    last_eval_step, last_save_step, last_selfplay_step = 0, 0, 0
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
                get_copy_of_self(training_monitor.run, agent, step),
                removable=cfg.get("self_removable", True),
            )
            if evaluate_against_self:
                evaluation_opponents.append(
                    get_copy_of_self(training_monitor.run, agent, step)
                )
            last_selfplay_step = step
            print(f"Added self-play agent to opponent pool at step {step}.")

        train_episodes_playing_start_time = time.time()
        opponent = opponent_pool.sample_opponent()

        steps_played = 0
        ppo_actions_counter = {agent.name: 0 for agent in agent.agents}
        for env in train_envs:
            episode = run_episode(
                env, agent, opponent, agent.agent_repeat, ppo_actions_counter
            )

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

        train_episodes_playing_duration = (
            time.time() - train_episodes_playing_start_time
        )

        # Update model
        update_start_time = time.time()
        train_metrics = agent.ppo.update(global_step=step)
        update_duration = time.time() - update_start_time

        # Log training metrics
        total_actions = sum(ppo_actions_counter.values())
        actions_percentages = {
            agent_name: count / max(total_actions, 1)
            for agent_name, count in ppo_actions_counter.items()
        }
        time_metrics = {
            "Time/train_episodes_playing": train_episodes_playing_duration,
            "Time/train_episodes_per_step": train_episodes_playing_duration
            / max(steps_played, 1),
            "Time/model_update": update_duration,
        }
        train_metrics = {f"Losses/{k}": v for k, v in train_metrics.items()}
        training_monitor.run.log(
            {**train_metrics, **time_metrics, **actions_percentages}, step=step
        )

        if cfg.save_model and (
            (step - last_save_step >= cfg.save_model_freq) or step >= cfg.train_steps
        ):
            agent.save_to_wandb(
                training_monitor.run, folder_name=f"checkpoint_step_{step}"
            )
            last_save_step = step

        # Evaluate agent periodically
        if step - last_eval_step >= cfg.eval_freq or step >= cfg.train_steps:
            eval_start_time = time.time()
            final_evaluation = step >= cfg.train_steps
            save_episodes_per_outcome = (
                cfg.video_episodes_per_outcome if final_evaluation else 0
            )
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

    if os.environ.get("DEBUG", "False").lower() == "true":
        CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "debug.yaml"
        os.environ["WANDB_MODE"] = "offline"

    with open(CONFIG_PATH, "r") as f:
        cfg = OmegaConf.load(f)
        train(cfg)
