import gym
import numpy as np
from slbo.envs import BaseModelBasedEnv


class Acrobot(BaseModelBasedEnv, gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.env = gym.make("Acrobot-v1").env

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
        done = self._done_func(next_states)
        reward = np.zeros_like(done, dtype=np.float32)
        reward[np.where(done==False)[0]] = -1
        return reward, done

    def _done_func(self, obs):
        cos_0, sin_0, cos_1, sin_1 = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
        done= -cos_0 - (cos_1 * cos_0 - sin_1 * sin_0) > 1.0
        return done
