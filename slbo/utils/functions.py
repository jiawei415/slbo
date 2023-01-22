import numpy as np
import lunzi.nn as nn
from slbo.utils.dataset import Dataset
from slbo.envs.batched_env import BatchedEnv
from slbo.utils.runner import Runner
from slbo.partial_envs import make_env


def format_data(data: dict, format=3):
    new_data: dict = {}
    for key, val in data.items():
        new_data[key] = round(val, format)
    return new_data


def evaluate(settings, n_test_samples):
    results = {}
    for runner, policy, name in settings:
        runner.reset()
        _, ep_infos = runner.run(policy, n_test_samples)
        returns = np.array([ep_info["return"] for ep_info in ep_infos])
        returns_mean, returns_std = 0, 0
        if len(returns) > 0:
            returns_mean, returns_std = np.mean(returns), np.std(returns) / len(returns)
        results.update(
            {
                f"{name}/reward": returns_mean,
                f"{name}/reward_std": returns_std,
                f"{name}/reward_len": len(returns),
            }
        )
    return results


def add_multi_step(src: Dataset, dst: Dataset):
    n_envs = 1
    dst.extend(src[:-n_envs])

    ending = src[-n_envs:].copy()
    ending.timeout = True
    dst.extend(ending)


def make_real_runner(n_envs, env_id, runner_config={}):

    batched_env = BatchedEnv([make_env(env_id) for _ in range(n_envs)])
    return Runner(batched_env, **runner_config)


def get_criterion(loss_type):
    criterion_map = {
        "L1": nn.L1Loss(),
        "L2": nn.L2Loss(),
        "MSE": nn.MSELoss(),
    }
    return criterion_map[loss_type]
