import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from slbo.envs import BaseModelBasedEnv


class Pendulum(BaseModelBasedEnv, gym.Env):
    def __init__(self, g=10.0, **kwargs) -> None:
        super().__init__()
        # self.env = gym.make("Pendulum-v0").env
        self.env = PendulumEnv(g=g)

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
        th, thdot = np.arccos(obs[:, 0]), obs[:, -1]
        u = np.clip(action.squeeze(), self.action_space.low, self.action_space.high)
        costs = (
            np.square(self._angle_normalize(th))
            + 0.1 * np.square(thdot)
            + 0.001 * np.square(u)
        )
        return -costs

    def _angle_normalize(self, x):
        return np.mod(x + np.pi, 2 * np.pi) - np.pi


class PendulumEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, g=10.0):
        self.g = g
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.viewer = None

        high = np.array([1.0, 1.0, self.max_speed])
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,)
        )
        self.observation_space = spaces.Box(low=-high, high=high)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        # g = 10.0
        m = 1.0
        l = 1.0
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        newthdot = (
            thdot
            + (-3 * self.g / (2 * l) * np.sin(th + np.pi) + 3.0 / (m * l**2) * u) * dt
        )
        newth = th + newthdot * dt
        newthdot = np.clip(
            newthdot, -self.max_speed, self.max_speed
        )  # pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode="human", close=False):
        pass


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
