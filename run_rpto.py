# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from slbo.utils.logger import configure
from slbo.utils.dataset import Dataset, gen_dtype
from slbo.utils.flags import FLAGS
from slbo.utils.OU_noise import OUNoise
from slbo.utils.normalizer import Normalizers
from slbo.utils.runner import Runner
from slbo.utils.tf_utils import get_tf_config
from slbo.utils.functions import (
    format_data,
    evaluate,
    add_multi_step,
    make_real_runner,
    get_criterion,
)
from slbo.policies.gaussian_mlp_policy import GaussianMLPPolicy
from slbo.policies.discrete_mlp_policy import DiscreteMLPPolicy
from slbo.v_function.mlp_v_function import MLPVFunction
from slbo.dynamics_model import DynamicsModel
from slbo.envs.virtual_env import VirtualEnv
from slbo.partial_envs import make_env
from slbo.algos.RPO import RPO
from slbo.algos.RTO import RTO

logger = configure(FLAGS.log_dir)


def main():
    FLAGS.set_seed()
    FLAGS.freeze()

    env = make_env(FLAGS.env.id, FLAGS.env.target_config)
    env.verify()

    dim_state = int(np.prod(env.observation_space.shape))
    if FLAGS.env.action_type == "continuous":
        dim_action = int(np.prod(env.action_space.shape))
    elif FLAGS.env.action_type == "discrete":
        dim_action = env.action_space.n

    dtype = gen_dtype(env, "state action next_state reward done timeout")
    train_set = Dataset(dtype, FLAGS.rollout.max_buf_size)
    dev_set = Dataset(dtype, FLAGS.rollout.max_buf_size)

    normalizers = Normalizers(dim_action=dim_action, dim_state=dim_state)
    noise = OUNoise(
        env.action_space,
        theta=FLAGS.OUNoise.theta,
        sigma=FLAGS.OUNoise.sigma,
        shape=(1, dim_action),
    )
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
    vfn = MLPVFunction(dim_state, [64, 64], normalizers.state)
    model = DynamicsModel(dim_state, dim_action, normalizers, FLAGS.model.hidden_sizes)

    criterion = get_criterion(FLAGS.model.loss)
    rto = RTO(
        dim_state=dim_state,
        dim_action=dim_action,
        vfn=vfn,
        model=model,
        normalizers=normalizers,
        criterion=criterion,
        action_type=FLAGS.env.action_type,
        **FLAGS.RTO.as_dict(),
    )
    n_update = (
        FLAGS.common.n_stages * FLAGS.common.n_iters * FLAGS.common.n_policy_iters
    )
    rpo = RPO(
        dim_state=dim_state,
        dim_action=dim_action,
        policy=policy,
        vfn=vfn,
        n_update=n_update,
        action_type=FLAGS.env.action_type,
        **FLAGS.RPO.as_dict(),
    )

    tf.get_default_session().run(tf.global_variables_initializer())

    saver = nn.ModuleDict({"policy": policy, "model": model, "vfn": vfn})
    logger.info(saver)
    if os.path.exists(FLAGS.model.model_load_path):
        model_load_path = os.path.join(FLAGS.model.model_load_path, FLAGS.config.env)
        for file in os.listdir(model_load_path):
            if "2023" in file:
                model_load_path = os.path.join(model_load_path, f"{file}/final.npy")
                break
        saver.load_state_dict(np.load(model_load_path, allow_pickle=True)[()])
        logger.info(f"Load model from {model_load_path}")

    virt_env = VirtualEnv(
        model,
        make_env(FLAGS.env.id),
        FLAGS.plan.n_envs,
        opt_model=FLAGS.common.opt_model,
    )
    virt_runner = Runner(
        virt_env, **{**FLAGS.runner.as_dict(), "max_steps": FLAGS.plan.max_steps}
    )

    runners = {
        "test": make_real_runner(4, FLAGS.env.id, FLAGS.runner.as_dict()),
        "dev": make_real_runner(1, FLAGS.env.id, FLAGS.runner.as_dict()),
        "collect": make_real_runner(1, FLAGS.env.id, FLAGS.runner.as_dict()),
        "source": virt_runner,
    }
    settings = [
        (runners["test"], policy, "Real_Env"),
        (runners["source"], policy, "Virt_Env"),
    ]

    # evaluation
    test_results = evaluate(settings, FLAGS.rollout.n_test_samples)
    logger.record("global/stage", 0)
    logger.record("global/env_step", 0)
    logger.record("global/update_step", 0)
    for key, val in test_results.items():
        logger.record(f"test/{key}", val)
    logger.dump(0)

    now_virt_step, now_real_step = 0, 0
    now_model_update, now_policy_step = 0, 0
    for T in range(1, FLAGS.common.n_stages + 1):
        # collect data in real env
        if FLAGS.env.action_type == "continuous":
            collect_policy = noise.make(policy)
        elif FLAGS.env.action_type == "discrete":
            collect_policy = policy
        recent_train_set, ep_infos = runners["collect"].run(
            collect_policy, FLAGS.rollout.n_train_samples
        )
        add_multi_step(recent_train_set, train_set)
        returns = np.array([ep_info["return"] for ep_info in ep_infos])
        returns_mean, returns_std = 0, 0
        if len(returns) > 0:
            returns_mean, returns_std = np.mean(returns), np.std(returns) / len(returns)
        train_real_results = {
            "Real_Env/reward": returns_mean,
            "Real_Env/reward_std": returns_std,
            "Real_Env/reward_len": len(returns),
        }
        now_real_step += FLAGS.rollout.n_train_samples

        if T == 1:  # check
            samples = train_set.sample_multi_step(100, 1, FLAGS.model.multi_step)
            for i in range(FLAGS.model.multi_step - 1):
                masks = 1 - (samples.done[i] | samples.timeout[i])[..., np.newaxis]
                assert np.allclose(
                    samples.state[i + 1] * masks, samples.next_state[i] * masks
                )
        if (
            FLAGS.rollout.normalizer == "policy"
            or FLAGS.rollout.normalizer == "uniform"
            and T == 1
        ):
            normalizers.state.update(recent_train_set.state)
            if FLAGS.env.action_type == "continuous":
                normalizers.action.update(recent_train_set.action)
            normalizers.diff.update(
                recent_train_set.next_state - recent_train_set.state
            )

        # collect data for dev
        if FLAGS.env.action_type == "continuous":
            dev_policy = noise.make(policy)
        elif FLAGS.env.action_type == "discrete":
            dev_policy = policy
        add_multi_step(
            runners["dev"].run(dev_policy, FLAGS.rollout.n_dev_samples)[0], dev_set
        )

        for i in range(FLAGS.common.n_iters):
            # update model
            for model_iter in range(FLAGS.common.n_model_iters):
                samples = train_set.sample_multi_step(
                    FLAGS.model.train_batch_size, 1, FLAGS.model.multi_step
                )
                rto_results = rto.train(
                    samples.state,
                    samples.next_state,
                    samples.action,
                    ~samples.done & ~samples.timeout,
                )
                now_model_update += 1
                logger.info(
                    f"[RTO]: {str(model_iter + 1).zfill(2)} {format_data(rto_results)}"
                )

            if i % FLAGS.model.validation_freq == 0:
                samples = dev_set.sample_multi_step(
                    FLAGS.model.train_batch_size, 1, FLAGS.model.multi_step
                )
                action = samples.action
                if FLAGS.env.action_type == "discrete":
                    action = action.astype(np.int64)
                    one_hot_action = np.zeros(samples.action.shape + (dim_action,))
                    for i in range(action.shape[0]):
                        one_hot_action[i, np.arange(action.shape[1]), action[0]] = 1
                    action = one_hot_action
                rto_loss, dy_loss = rto.get_rto_loss(
                    samples.state,
                    samples.next_state,
                    action,
                    ~samples.done & ~samples.timeout,
                )
                if np.isnan(rto_loss) or np.isnan(dy_loss):
                    logger.info(
                        f"# Iter {str(i).zfill(2)}: Loss = [rto nan = {np.isnan(rto_loss)}, sp nan = {np.isnan(dy_loss)}],"
                        f" after {now_model_update} steps."
                    )

            # update policy
            for policy_iter in range(FLAGS.common.n_policy_iters):
                if FLAGS.algorithm != "MF" and FLAGS.common.start == "buffer":
                    runners["source"].set_state(
                        train_set.sample(FLAGS.plan.n_envs).state
                    )
                else:
                    runners["source"].reset()

                # collect source(model) env data
                data, ep_infos = runners["source"].run(
                    policy, FLAGS.plan.n_policy_samples
                )
                returns = [info["return"] for info in ep_infos]
                returns_mean, returns_std = 0, 0
                if len(returns) > 0:
                    returns_mean, returns_std = np.mean(returns), np.std(returns) / len(
                        returns
                    )
                train_virt_results = {
                    "Virt_Env/reward": returns_mean,
                    "Virt_Env/reward_std": returns_std,
                    "Virt_Env/reward_len": len(returns),
                }
                now_virt_step += FLAGS.plan.n_policy_samples

                # collect target(real) env data
                tar_data, ep_infos = runners["collect"].run(
                    policy, FLAGS.plan.n_policy_samples
                )
                add_multi_step(tar_data, train_set)
                now_real_step += FLAGS.plan.n_policy_samples

                advantages, values = runners["source"].compute_advantage(vfn, data)
                tar_advantages, tar_values = runners["collect"].compute_advantage(
                    vfn, tar_data
                )
                rpo_results = rpo.train(
                    data,
                    advantages,
                    values,
                    tar_data,
                    tar_advantages,
                    tar_values,
                    now_policy_step,
                )
                now_policy_step += 1
                logger.info(
                    f"[RPO]: {str(policy_iter + 1).zfill(2)} {format_data(rpo_results)}"
                )

        test_results = evaluate(settings, FLAGS.rollout.n_test_samples)

        logger.record("global/stage", T)
        logger.record("global/env_step", now_real_step)
        logger.record("global/update_step", now_policy_step)
        for key, val in test_results.items():
            logger.record(f"test/{key}", val)
        for key, val in train_real_results.items():
            logger.record(f"train/{key}", val)
        for key, val in train_virt_results.items():
            logger.record(f"train/{key}", val)
        for key, val in rpo_results.items():
            logger.record(f"update/{key}", val)
        for key, val in rto_results.items():
            logger.record(f"update/{key}", val)
        logger.dump(T)

        if T % FLAGS.ckpt.n_save_stages == 0:
            np.save(f"{FLAGS.log_dir}/stage_{str(T).zfill(3)}", saver.state_dict())

    np.save(f"{FLAGS.log_dir}/final", saver.state_dict())


if __name__ == "__main__":
    with tf.Session(config=get_tf_config()):
        main()
