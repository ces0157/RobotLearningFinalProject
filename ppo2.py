from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common import utils
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import torch as th

from stable_baselines3.common import logger
import gymnasium as gym
import craftium
import numpy as np


#env = gym.make("Craftium/Speleo-v0", obs_width = 512, obs_height = 512, frameskip=2)
env = make_vec_env("Craftium/ChopTree-v0", n_envs=4)
#env = DummyVecEnv([lambda: gym.make("Craftium/Speleo-v0", obs_width = 512, obs_height = 512, frameskip=2)])
#env = VecNormalize(env, norm_obs=False, norm_reward=True, n_envs=4)
#env = gym.make("Craftium/Speleo-v0", obs_width = 512, obs_height = 512, frameskip=2)
# env.mouse_mov = .25

# print(env.observation_space)
# net_arch = [
#     dict(pi=[32, 64, 64], vf=[64, 64]),  # Policy and value function architectures
# ]

#policy = ActorCriticCnnPolicy(env.observation_space, env.action_space, net_arch=net_arch, lr_schedule=utils)

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 128], vf=[128, 128]))

model = PPO("CnnPolicy", env=env, verbose=1, device="cuda", policy_kwargs=policy_kwargs, n_steps=2048, batch_size=256, n_epochs=30)
model.set_parameters("Tree_really_good")
new_logger = logger.configure("Tree2", ["stdout", "csv"])
model.set_logger(new_logger)
model.learn(total_timesteps=100000, progress_bar=True)
model.save("Tree_new")

# env.close()