# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import gym
import numpy as np
from slbo.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from slbo.envs.mujoco.walker2d_env import Walker2DEnv
from slbo.envs.mujoco.humanoid_env import HumanoidEnv
from slbo.envs.mujoco.ant_env import AntEnv
from slbo.envs.mujoco.hopper_env import HopperEnv
from slbo.envs.mujoco.swimmer_env import SwimmerEnv
from slbo.envs.mujoco.cartpole import CartPole
from slbo.envs.mujoco.pendulum import Pendulum
from slbo.envs.mujoco.acrobot import Acrobot
from slbo.envs.mujoco.mountaincarcontinuous import MountainCarContinuous


def make_env(id: str, env_config: dict = {}):
    envs = {
        "HalfCheetah-v2": HalfCheetahEnv,
        "Walker2D-v2": Walker2DEnv,
        "Humanoid-v2": HumanoidEnv,
        "Ant-v2": AntEnv,
        "Hopper-v2": HopperEnv,
        "Swimmer-v2": SwimmerEnv,
        "CartPole-v0": CartPole,
        "Pendulum-v0": Pendulum,
        "Acrobot-v1": Acrobot,
        "MountainCarContinuous-v0": MountainCarContinuous,
    }
    if id in envs.keys():
        env = envs[id](**env_config)
    else:
        env = gym.make(id)
    if not hasattr(env, "reward_range"):
        env.reward_range = (-np.inf, np.inf)
    if not hasattr(env, "metadata"):
        env.metadata = {}
    env.seed(np.random.randint(2**60))
    return env
