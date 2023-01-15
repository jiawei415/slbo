# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


from gym.envs.registration import register

register(
    id="CartPole-v5",
    entry_point="slbo.envs.mujoco.cartpole:RiverSwimRandom",
)
