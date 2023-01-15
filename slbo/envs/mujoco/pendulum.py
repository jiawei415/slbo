import gym
import numpy as np
from slbo.envs import BaseModelBasedEnv


class Pendulum(BaseModelBasedEnv, gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.env = gym.make("Pendulum-v0").env

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def seed(self, seed):
        return self.env.seed(seed)

    def mb_step(self, states, actions, next_states):
        reward = self._reward_func(states, actions)
        return reward, np.zeros(states.shape[0]).astype(np.bool)

    def _reward_func(self, obs, action):
        th, thdot = np.arccos(obs[:, 0]), obs[:,-1]
        u = np.clip(action.squeeze(), self.action_space.low, self.action_space.high)
        costs = (
            np.square(self._angle_normalize(th))
            + 0.1 * np.square(thdot)
            + 0.001 * np.square(u)
        )
        return -costs

    def _angle_normalize(self, x):
        return np.mod(x + np.pi, 2 * np.pi) - np.pi
