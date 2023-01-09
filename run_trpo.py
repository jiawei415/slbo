# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from slbo.utils.logger import configure
from slbo.utils.flags import FLAGS
from slbo.utils.normalizer import Normalizers
from slbo.utils.tf_utils import get_tf_config
from slbo.utils.runner import Runner
from slbo.policies.gaussian_mlp_policy import GaussianMLPPolicy
from slbo.v_function.mlp_v_function import MLPVFunction
from slbo.partial_envs import make_env
from slbo.algos.TRPO import TRPO

logger = configure(FLAGS.log_dir)


def format_data(data: dict, format=3):
    new_data: dict = {}
    for key, val in data.items():
        new_data[key] = round(val, format)
    return new_data


def evaluate(settings):
    results = {}
    for runner, policy, name in settings:
        runner.reset()
        _, ep_infos = runner.run(policy, FLAGS.rollout.n_test_samples)
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


def make_real_runner(n_envs):
    from slbo.envs.batched_env import BatchedEnv

    batched_env = BatchedEnv([make_env(FLAGS.env.id) for _ in range(n_envs)])
    return Runner(batched_env, rescale_action=True, **FLAGS.runner.as_dict())


def main():
    FLAGS.set_seed()
    FLAGS.freeze()

    env = make_env(FLAGS.env.id)
    dim_state = int(np.prod(env.observation_space.shape))
    dim_action = int(np.prod(env.action_space.shape))

    env.verify()

    normalizers = Normalizers(dim_action=dim_action, dim_state=dim_state)

    policy = GaussianMLPPolicy(
        dim_state, dim_action, normalizer=normalizers.state, **FLAGS.policy.as_dict()
    )
    vfn = MLPVFunction(dim_state, [64, 64], normalizers.state)
    algo = TRPO(
        vfn=vfn,
        policy=policy,
        dim_state=dim_state,
        dim_action=dim_action,
        **FLAGS.TRPO.as_dict(),
    )

    tf.get_default_session().run(tf.global_variables_initializer())

    runners = {
        "test": make_real_runner(4),
        "train": make_real_runner(1),
    }
    settings = [(runners["test"], policy, "Real Env")]

    saver = nn.ModuleDict({"policy": policy, "vfn": vfn})
    logger.info(saver)

    # evaluation
    test_results = evaluate(settings)
    logger.record("global/stage", 0)
    logger.record("global/env_step", 0)
    logger.record("global/update_step", 0)
    for key, val in test_results.items():
        logger.record(f"test/{key}", val)
    logger.dump(0)

    max_ent_coef = FLAGS.TRPO.ent_coef

    runners["train"].reset()
    now_update, now_step = 0, 0
    for T in range(1, FLAGS.slbo.n_stages + 1):
        if T == 50:
            max_ent_coef = 0.0

        for i in range(FLAGS.slbo.n_iters):
            for policy_iter in range(FLAGS.slbo.n_policy_iters):
                data, ep_infos = runners["train"].run(
                    policy, FLAGS.rollout.n_train_samples
                )
                returns = [info["return"] for info in ep_infos]
                returns_mean, returns_std = 0, 0
                if len(returns) > 0:
                    returns_mean, returns_std = np.mean(returns), np.std(returns) / len(
                        returns
                    )
                train_results = {
                    "reward": returns_mean,
                    "reward_std": returns_std,
                    "reward_len": len(returns),
                }
                now_step += FLAGS.rollout.n_train_samples

                normalizers.state.update(data.state)
                advantages, values = runners["train"].compute_advantage(vfn, data)
                update_results = algo.train(max_ent_coef, data, advantages, values)
                now_update += 1

                logger.info(
                    f"[TRPO]: {str(policy_iter + 1).zfill(2)} {format_data(train_results)} {format_data(update_results)}"
                )

        test_results = evaluate(settings)

        logger.record("global/stage", T)
        logger.record("global/env_step", now_step)
        logger.record("global/update_step", now_update)
        for key, val in test_results.items():
            logger.record(f"test/{key}", val)
        for key, val in train_results.items():
            logger.record(f"train/{key}", val)
        for key, val in update_results.items():
            logger.record(f"update/{key}", val)
        logger.dump(T)

        if T % FLAGS.ckpt.n_save_stages == 0:
            np.save(f"{FLAGS.log_dir}/stage_{str(T).zfill(3)}", saver.state_dict())

    np.save(f"{FLAGS.log_dir}/final", saver.state_dict())


if __name__ == "__main__":
    with tf.Session(config=get_tf_config()):
        main()
