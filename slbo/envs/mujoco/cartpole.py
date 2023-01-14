import gym
import numpy as np
from slbo.envs import BaseModelBasedEnv


class CartPole(BaseModelBasedEnv, gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.env = gym.make("CartPole-v0")
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4

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
        reward = np.ones(states.shape[0])
        done = self._done_func(next_states)
        return reward, done

    def _done_func(self, obs):
        x, theta = obs[:, 0], obs[:, 2]
        x_done = (x < -self.x_threshold) | (x > self.x_threshold)
        theta_done = (theta < -self.theta_threshold_radians) | (
            theta > self.theta_threshold_radians
        )
        done = x_done | theta_done
        return done
