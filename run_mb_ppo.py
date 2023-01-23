# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
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
from slbo.dynamics_model import DynamicsModel
from slbo.v_function.mlp_v_function import MLPVFunction
from slbo.envs.virtual_env import VirtualEnv
from slbo.partial_envs import make_env
from slbo.algos.PPO import PPO
from slbo.algos.MULTISTEP import MultiStep

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
    multi_step = MultiStep(
        dim_state=dim_state,
        dim_action=dim_action,
        model=model,
        normalizers=normalizers,
        criterion=criterion,
        action_type=FLAGS.env.action_type,
        **FLAGS.MultiStep.as_dict(),
    )
    n_update = (
        FLAGS.common.n_stages * FLAGS.common.n_iters * FLAGS.common.n_policy_iters
    )
    ppo = PPO(
        vfn=vfn,
        policy=policy,
        dim_state=dim_state,
        dim_action=dim_action,
        n_update=n_update,
        action_type=FLAGS.env.action_type,
        **FLAGS.PPO.as_dict(),
    )

    tf.get_default_session().run(tf.global_variables_initializer())

    saver = nn.ModuleDict({"policy": policy, "model": model, "vfn": vfn})
    logger.info(saver)

    virt_env = VirtualEnv(
        model,
        make_env(FLAGS.env.id),
        FLAGS.plan.n_envs,
        opt_model=FLAGS.common.opt_model,
    )
    virt_runner = Runner(
        virt_env,
        **{
            **FLAGS.runner.as_dict(),
            "max_steps": FLAGS.plan.max_steps,
            "rescale_action": False,
        },
    )

    runners = {
        "test": make_real_runner(4, FLAGS.env.id, FLAGS.runner.as_dict()),
        "dev": make_real_runner(1, FLAGS.env.id, FLAGS.runner.as_dict()),
        "collect": make_real_runner(1, FLAGS.env.id, FLAGS.runner.as_dict()),
        "train": virt_runner,
    }
    settings = [
        (runners["test"], policy, "Real_Env"),
        (runners["train"], policy, "Virt_Env"),
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

        # recent_states = obsvs
        # ref_actions = policy.eval('actions_mean actions_std', states=recent_states)
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
                model_results = multi_step.train(
                    samples.state,
                    samples.next_state,
                    samples.action,
                    ~samples.done & ~samples.timeout,
                )
                now_model_update += 1
                logger.info(
                    f"[MODEL]: {str(model_iter + 1).zfill(2)} {format_data(model_results)}"
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
                loss = multi_step.get_loss(
                    samples.state,
                    samples.next_state,
                    action,
                    ~samples.done & ~samples.timeout,
                )
                if np.isnan(loss):
                    logger.info(
                        f"# Iter {str(i).zfill(2)}: Loss = [model nan = {np.isnan(loss)}],"
                        f" after {now_model_update} steps."
                    )

            # update policy
            for policy_iter in range(FLAGS.common.n_policy_iters):
                if FLAGS.algorithm != "MF" and FLAGS.common.start == "buffer":
                    runners["train"].set_state(
                        train_set.sample(FLAGS.plan.n_envs).state
                    )
                else:
                    runners["train"].reset()

                data, ep_infos = runners["train"].run(
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

                advantages, values = runners["train"].compute_advantage(vfn, data)
                policy_results = ppo.train(data, advantages, values, now_policy_step)
                now_policy_step += 1
                logger.info(
                    f"[PPO]: {str(policy_iter + 1).zfill(2)} {format_data(policy_results)}"
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
        for key, val in policy_results.items():
            logger.record(f"update/{key}", val)
        for key, val in model_results.items():
            logger.record(f"update/{key}", val)
        logger.dump(T)

        if T % FLAGS.ckpt.n_save_stages == 0:
            np.save(f"{FLAGS.log_dir}/stage_{str(T).zfill(3)}", saver.state_dict())

    np.save(f"{FLAGS.log_dir}/final", saver.state_dict())


if __name__ == "__main__":
    with tf.Session(config=get_tf_config()):
        main()
