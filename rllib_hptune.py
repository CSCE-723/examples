"""
Train a policy for the TankEnv using Ray RLlib PPO and Ray Tune Tuner API with
Weights & Biases logging.
"""

import os
import pathlib

from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.schedulers import ASHAScheduler

N_TRIALS = 10
ENV_ID = "CartPole-v1"
LOGDIR = pathlib.Path("./rllib_logs").absolute()
MAX_ITERATIONS = 100
MAX_REWARD = 300

if __name__ == "__main__":
    log_dir = pathlib.Path(LOGDIR).absolute()
    os.makedirs(log_dir, exist_ok=True)
    # Stop when we've either reached 100 training iterations or reward=300
    stopping_criteria = {
        "training_iteration": MAX_ITERATIONS,
        "episode_reward_mean": MAX_REWARD
        }

    asha = ASHAScheduler(
        metric="env_runners/episode_return_mean",
        mode="max",
        max_t=100,
        grace_period=10,
        reduction_factor=3,
    )

    config = (
        PPOConfig()
        .environment(ENV_ID)
        .env_runners(num_env_runners=4)
        .training(
            # These params are fixed for all trials.
            kl_coeff=1.0,
            lambda_=0.95,
            clip_param=0.2,
            # These params are randomly drawn from a set.
            lr=tune.loguniform(1e-5, 1e-3),
            num_epochs=tune.choice([10, 20, 30]),
            minibatch_size=tune.choice([128, 512, 2048]),
            train_batch_size_per_learner=tune.choice([10000, 20000, 40000]),
        )
    )

    tuner = tune.Tuner(
        config.algo_class,
        tune_config=tune.TuneConfig(
            scheduler=asha,
            num_samples=N_TRIALS,
        ),
        param_space=config,
        run_config=tune.RunConfig(
            stop=stopping_criteria,
            storage_path=log_dir,
            # callbacks=[WandbLoggerCallback(project="cartpole_rllib", log_config=True)],
            ),
    )
    results = tuner.fit()
