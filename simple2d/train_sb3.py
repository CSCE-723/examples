"""
Train a policy for the TankEnv using stable_baselines3 PPO with
Weights & Biases, logging, periodic checkpointing, and vectorized envs.
"""
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env
from gym_env.gym_env import Simple2DEnv
from stable_baselines3.common.logger import configure
from wandb.integration.sb3 import WandbCallback
import wandb

# Run settings:
total_timesteps = 1_000_000  # Total training steps

# Set up directories
log_dir = "./sb3_logs"
ckpt_dir = os.path.join(log_dir, "checkpoints")
wandb_dir = os.path.join(log_dir, "wandb")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(wandb_dir, exist_ok=True)

# Initialize wandb run before training
wandb_run = wandb.init(
    project="simple2d_sb3",
    name="ppo_simple2d_run",
    dir=wandb_dir,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
)

# Only initialize wandb in the main process, let WandbCallback handle run logic
# (Do not call wandb.init() here)

# Create vectorized environment (10 parallel envs) using make_vec_env
num_envs = 10
env = make_vec_env(
    Simple2DEnv,
    n_envs=num_envs,
    env_kwargs={'config':{"render_mode": None}},
    vec_env_cls=None,  # Use default (SubprocVecEnv if n_envs > 1)
    monitor_dir=log_dir,  # Needed for video recording
)

# Optionally wrap with VecVideoRecorder for periodic video saving
# This can be very slow, especially with our rough rendering in the example
# Recommend not to use in training, only in evaluation
# env = VecVideoRecorder(
#     env,
#     video_folder=os.path.join(log_dir, "videos"),
#     record_video_trigger=lambda x: x % 5000 == 0,
#     video_length=500,
#     name_prefix="ppo-simple2d-vid",
# )

# Optional: check environment compliance (single env)
check_env(Simple2DEnv(), warn=True)

# Set up SB3 logger to use TensorBoard (for wandb sync)
new_logger = configure(log_dir, ['stdout', 'tensorboard'])

# Set up checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=max(10_000 // num_envs, 1),  # Save every 10k steps
    save_path=ckpt_dir,
    name_prefix="ppo_simple2d",
    # save_replay_buffer=True,
    # save_vecnormalize=True,
)

# Set up WandbCallback for hyperparam/model saving and video upload
wandb_callback = WandbCallback(
    # gradient_save_freq=1000,
    model_save_path=os.path.join(log_dir, "wandb_models"),
    verbose=2,
    log="all",
)

# Instantiate the agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    device='cpu',  # NOTE: comment out this line to use GPU if available
)
model.set_logger(new_logger)

# Train the agent
model.learn(
    total_timesteps=total_timesteps,
    callback=[checkpoint_callback, wandb_callback],
    progress_bar=True,
)

# Save the final model
model.save(os.path.join(log_dir, "ppo_simple2d_final"))
print(f"Model saved as {os.path.join(log_dir, 'ppo_simple2d_final.zip')}")

# Finish wandb run if started
if wandb.run is not None:
    wandb.finish()
