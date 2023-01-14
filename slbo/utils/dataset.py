# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import lunzi.dataset as dataset
import gym
from gym import spaces


def gen_dtype(env: gym.Env, fields: str):
    dtypes = {
        "state": ("state", "f8", env.observation_space.shape),
        # "action": ("action", "f8", env.action_space.shape),
        "next_state": ("next_state", "f8", env.observation_space.shape),
        "reward": ("reward", "f8"),
        "done": ("done", "bool"),
        "timeout": ("timeout", "bool"),
        "return_": ("return_", "f8"),
        "advantage": ("advantage", "f8"),
    }
    if isinstance(env.action_space, spaces.Discrete):
        dtypes.update({"action": ("action", "f8")})
    else:
        dtypes.update({"action": ("action", "f8", env.action_space.shape)})
    return [dtypes[field] for field in fields.split(" ")]


class Dataset(dataset.Dataset):
    def sample_multi_step(self, size: int, n_env: int, n_step=1):
        starts = np.random.randint(0, self._len, size=size)
        batch = []
        for step in range(n_step):
            batch.append(self[(starts + step * n_env) % self._len])
        return np.concatenate(batch).reshape(n_step, size).view(np.recarray)
