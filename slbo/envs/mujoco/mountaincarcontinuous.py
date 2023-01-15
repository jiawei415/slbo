import gym
import numpy as np
from slbo.envs import BaseModelBasedEnv


class MountainCarContinuous(BaseModelBasedEnv, gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.env = gym.make("MountainCarContinuous-v0").env
        self.goal_position = 0.45

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
        position = next_states[:, 0]
        done = position >= self.goal_position
        reward = np.zeros_like(done, dtype=np.float32)
        reward[np.where(done == True)[0]] = 100.0
        a = np.square(actions[: 0])
        reward -= np.square(actions[:, 0]) * 0.1
        return reward, done
