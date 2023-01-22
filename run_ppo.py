# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from slbo.utils.logger import configure
from slbo.utils.tf_utils import get_tf_config
from slbo.utils.flags import FLAGS
from slbo.utils.normalizer import Normalizers
from slbo.utils.functions import (
    format_data,
    evaluate,
    make_real_runner,
)
from slbo.policies.gaussian_mlp_policy import GaussianMLPPolicy
from slbo.policies.discrete_mlp_policy import DiscreteMLPPolicy
from slbo.v_function.mlp_v_function import MLPVFunction
from slbo.partial_envs import make_env
from slbo.algos.PPO import PPO

logger = configure(FLAGS.log_dir)


def main():
    FLAGS.set_seed()
    FLAGS.freeze()

    env = make_env(FLAGS.env.id, FLAGS.env.source_config)
    env.verify()

    dim_state = int(np.prod(env.observation_space.shape))
    if FLAGS.env.action_type == "continuous":
        dim_action = int(np.prod(env.action_space.shape))
    elif FLAGS.env.action_type == "discrete":
        dim_action = env.action_space.n

    normalizers = Normalizers(dim_action=dim_action, dim_state=dim_state)

    if FLAGS.env.action_type == "continuous":
        Policy = GaussianMLPPolicy
    elif FLAGS.env.action_type == "discrete":
        Policy = DiscreteMLPPolicy
    policy = Policy(
        dim_state,
        dim_action,
        normalizer=normalizers.state,
        **FLAGS.policy.as_dict(),
    )
    vfn = MLPVFunction(dim_state, hidden_sizes=[64, 64], normalizer=normalizers.state)

    n_update = (
        FLAGS.common.n_stages * FLAGS.common.n_iters * FLAGS.common.n_policy_iters
    )
    algo = PPO(
        vfn=vfn,
        policy=policy,
        dim_state=dim_state,
        dim_action=dim_action,
        n_update=n_update,
        action_type=FLAGS.env.action_type,
        **FLAGS.PPO.as_dict(),
    )

    tf.get_default_session().run(tf.global_variables_initializer())

    saver = nn.ModuleDict({"policy": policy, "vfn": vfn})
    logger.info(saver)

    runners = {
        "test": make_real_runner(4, FLAGS.env.id, FLAGS.runner.as_dict()),
        "train": make_real_runner(1, FLAGS.env.id, FLAGS.runner.as_dict()),
    }
    settings = [(runners["test"], policy, "Real_Env")]

    # evaluation
    test_results = evaluate(settings, FLAGS.rollout.n_test_samples)
    logger.record("global/stage", 0)
    logger.record("global/env_step", 0)
    logger.record("global/update_step", 0)
    for key, val in test_results.items():
        logger.record(f"test/{key}", val)
    logger.dump(0)

    runners["train"].reset()
    now_update, now_step = 0, 0
    for T in range(1, FLAGS.common.n_stages + 1):
        for i in range(FLAGS.common.n_iters):
            for policy_iter in range(FLAGS.common.n_policy_iters):
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
                update_results = algo.train(data, advantages, values, now_update)
                now_update += 1

                logger.info(
                    f"[PPO]: {str(policy_iter + 1).zfill(2)} {format_data(train_results)} {format_data(update_results)}"
                )

        test_results = evaluate(settings, FLAGS.rollout.n_test_samples)

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
