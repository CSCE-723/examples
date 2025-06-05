"""
Train a policy for the TankEnv using Ray RLlib PPO and Ray Tune Tuner API with
Weights & Biases logging.
"""
import os
import pathlib
from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig
from gym_env.gym_env import Simple2DEnv
import ray
from ray.tune.registry import register_env
from ray.air.integrations.wandb import WandbLoggerCallback


def env_creator(env_config):
    return Simple2DEnv(env_config)


register_env("Simple2D", env_creator)

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    log_dir = pathlib.Path("./rllib_logs").absolute()
    os.makedirs(log_dir, exist_ok=True)

    # Build PPOConfig
    config = (
        PPOConfig()
        .environment(env="Simple2D")
        .framework("torch")
        .env_runners(num_env_runners=10)
        .training(lr=1e-4)
    )

    tuner = tune.Tuner(
        config.algo_class,
        param_space=config,
        run_config=train.RunConfig(
            storage_path=log_dir,
            stop={"training_iteration": 10},
            callbacks=[
                WandbLoggerCallback(
                    project="simple2d_rllib",
                    name="ppo_simple2d_rllib_run",
                    log_config=True,
                    save_checkpoints=True,
                    upload_checkpoints=True,
                )
            ],
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=5,
                checkpoint_at_end=True,
            ),
        ),
    )
    tuner.fit()
    ray.shutdown()
