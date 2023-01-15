# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
from collections import deque
import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from slbo.utils.logger import configure
from slbo.utils.average_meter import AverageMeter
from slbo.utils.flags import FLAGS
from slbo.utils.dataset import Dataset, gen_dtype
from slbo.utils.OU_noise import OUNoise
from slbo.utils.normalizer import Normalizers
from slbo.utils.tf_utils import get_tf_config
from slbo.utils.runner import Runner
from slbo.policies.gaussian_mlp_policy import GaussianMLPPolicy
from slbo.policies.discrete_mlp_policy import DiscreteMLPPolicy
from slbo.envs.virtual_env import VirtualEnv
from slbo.dynamics_model import DynamicsModel
from slbo.v_function.mlp_v_function import MLPVFunction
from slbo.partial_envs import make_env
from slbo.algos.RTPO import RTPO

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


def add_multi_step(src: Dataset, dst: Dataset):
    n_envs = 1
    dst.extend(src[:-n_envs])

    ending = src[-n_envs:].copy()
    ending.timeout = True
    dst.extend(ending)


def make_real_runner(n_envs):
    from slbo.envs.batched_env import BatchedEnv

    batched_env = BatchedEnv([make_env(FLAGS.env.id) for _ in range(n_envs)])
    return Runner(
        batched_env,
        rescale_action=bool(FLAGS.env.action_type == "continuous"),
        **FLAGS.runner.as_dict(),
    )


def main():
    FLAGS.set_seed()
    FLAGS.freeze()

    env = make_env(FLAGS.env.id)
    dim_state = int(np.prod(env.observation_space.shape))
    if FLAGS.env.action_type == "continuous":
        dim_action = int(np.prod(env.action_space.shape))
    elif FLAGS.env.action_type == "discrete":
        dim_action = env.action_space.n

    env.verify()

    normalizers = Normalizers(dim_action=dim_action, dim_state=dim_state)
    # batched noises
    noise = OUNoise(
        env.action_space,
        theta=FLAGS.OUNoise.theta,
        sigma=FLAGS.OUNoise.sigma,
        shape=(1, dim_action),
    )

    dtype = gen_dtype(env, "state action next_state reward done timeout")
    train_set = Dataset(dtype, FLAGS.rollout.max_buf_size)
    dev_set = Dataset(dtype, FLAGS.rollout.max_buf_size)

    if FLAGS.env.action_type == "continuous":
        policy = GaussianMLPPolicy(
            dim_state,
            dim_action,
            normalizer=normalizers.state,
            **FLAGS.policy.as_dict(),
        )
    elif FLAGS.env.action_type == "discrete":
        policy = DiscreteMLPPolicy(
            dim_state,
            dim_action,
            normalizer=normalizers.state,
            **FLAGS.policy.as_dict(),
        )
    vfn = MLPVFunction(dim_state, [64, 64], normalizers.state)
    model = DynamicsModel(dim_state, dim_action, normalizers, FLAGS.model.hidden_sizes)

    virt_env = VirtualEnv(
        model, make_env(FLAGS.env.id), FLAGS.plan.n_envs, opt_model=FLAGS.slbo.opt_model
    )
    virt_runner = Runner(
        virt_env, **{**FLAGS.runner.as_dict(), "max_steps": FLAGS.plan.max_steps}
    )

    criterion_map = {
        "L1": nn.L1Loss(),
        "L2": nn.L2Loss(),
        "MSE": nn.MSELoss(),
    }
    criterion = criterion_map[FLAGS.model.loss]
    n_update = FLAGS.slbo.n_stages * FLAGS.slbo.n_iters * FLAGS.slbo.n_policy_iters
    algo = RTPO(
        vfn=vfn,
        policy=policy,
        model=model,
        criterion=criterion,
        normalizers=normalizers,
        dim_state=dim_state,
        dim_action=dim_action,
        n_update=n_update,
        action_type=FLAGS.env.action_type,
        **FLAGS.RTPO.as_dict(),
    )

    tf.get_default_session().run(tf.global_variables_initializer())

    runners = {
        "test": make_real_runner(4),
        "collect": make_real_runner(1),
        "dev": make_real_runner(1),
        "target": make_real_runner(FLAGS.plan.n_envs),
        "train": virt_runner,
    }
    settings = [
        (runners["test"], policy, "Real_Env"),
        (runners["train"], policy, "Virt_Env"),
    ]

    if FLAGS.model.model_load_path is not None:
        model_load_path = os.path.join(FLAGS.model.model_load_path, FLAGS.config.env)
        for file in os.listdir(model_load_path):
            if "2023" in file:
                model_load_path = os.path.join(model_load_path, f"{file}/final.npy")
                break
        model.load_state_dict(
            np.load(model_load_path, allow_pickle=True).tolist()["model"]
        )
        logger.info("Load model from %s", model_load_path)
    saver = nn.ModuleDict({"policy": policy, "model": model, "vfn": vfn})
    logger.info(saver)

    # evaluation
    test_results = evaluate(settings)
    logger.record("global/stage", 0)
    logger.record("global/env_step", 0)
    logger.record("global/update_step", 0)
    for key, val in test_results.items():
        logger.record(f"test/{key}", val)
    logger.dump(0)

    now_virt_step, now_real_step = 0, 0
    now_model_update, now_policy_step = 0, 0
    for T in range(1, FLAGS.slbo.n_stages + 1):
        # collect data in real env
        if FLAGS.env.action_type == "continuous":
            recent_train_set, ep_infos = runners["collect"].run(
                noise.make(policy), FLAGS.rollout.n_train_samples
            )
        elif FLAGS.env.action_type == "discrete":
            recent_train_set, ep_infos = runners["collect"].run(
                policy, FLAGS.rollout.n_train_samples
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
            add_multi_step(
                runners["dev"].run(noise.make(policy), FLAGS.rollout.n_dev_samples)[0],
                dev_set,
            )
        elif FLAGS.env.action_type == "discrete":
            add_multi_step(
                runners["dev"].run(policy, FLAGS.rollout.n_dev_samples)[0], dev_set
            )

        for i in range(FLAGS.slbo.n_iters):
            # update model
            dy_losses = deque(maxlen=FLAGS.slbo.n_model_iters)
            rto_losses = deque(maxlen=FLAGS.slbo.n_model_iters)
            model_grad_norm_meter = AverageMeter()
            n_model_iters = FLAGS.slbo.n_model_iters
            for model_iter in range(n_model_iters):
                samples = train_set.sample_multi_step(
                    FLAGS.model.train_batch_size, 1, FLAGS.model.multi_step
                )
                rto_loss, dy_loss, model_grad_norm = algo.train_rto(
                    samples.state,
                    samples.next_state,
                    samples.action,
                    ~samples.done & ~samples.timeout,
                )
                dy_losses.append(dy_loss)
                rto_losses.append(rto_loss)
                model_grad_norm_meter.update(model_grad_norm)
                now_model_update += 1

            update_model_results = {
                "model/loss_dy": np.mean(dy_losses),
                "model/loss_rto": np.mean(rto_losses),
                "model/grad_norm": model_grad_norm_meter.get(),
            }

            if i % FLAGS.model.validation_freq == 0:
                samples = dev_set.sample_multi_step(
                    FLAGS.model.train_batch_size, 1, FLAGS.model.multi_step
                )
                action = samples.action
                if FLAGS.env.action_type == "discrete":
                    one_hot_action = np.zeros(samples.action.shape + (dim_action,))
                    for i in range(action.shape[0]):
                        one_hot_action[
                            i, np.arange(action.shape[1]), action[0].astype(np.int64)
                        ] = 1
                    action = one_hot_action
                rto_loss, dy_loss = algo.get_rto_loss(
                    samples.state,
                    samples.next_state,
                    action,
                    ~samples.done & ~samples.timeout,
                )
                if np.isnan(rto_loss) or np.isnan(np.mean(rto_losses)):
                    logger.info(
                        f"# Iter {str(i).zfill(2)}: Loss = [train nan = {np.isnan(np.mean(rto_losses))}, dev = {np.isnan(rto_loss)}], "
                        f"after {now_model_update} steps, grad_norm = {model_grad_norm_meter.get():.6f}"
                    )
                logger.info(
                    f"# Iter {str(i).zfill(2)}: Loss = [train = {np.mean(rto_losses):.3f}, dev = {rto_loss:.3f}], "
                    f"after {now_model_update} steps, grad_norm = {model_grad_norm_meter.get():.6f}"
                )

            # update policy
            for policy_iter in range(FLAGS.slbo.n_policy_iters):
                if FLAGS.algorithm != "MF" and FLAGS.slbo.start == "buffer":
                    runners["train"].set_state(
                        train_set.sample(FLAGS.plan.n_envs).state
                    )
                else:
                    runners["train"].reset()

                data, ep_infos = runners["train"].run(policy, FLAGS.plan.n_trpo_samples)
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
                now_virt_step += FLAGS.plan.n_trpo_samples

                runners["target"].reset()
                tar_data, ep_infos = runners["target"].run(
                    policy, FLAGS.plan.n_trpo_samples
                )
                now_real_step += FLAGS.plan.n_trpo_samples

                advantages, values = runners["train"].compute_advantage(vfn, data)
                tar_advantages, tar_values = runners["target"].compute_advantage(
                    vfn, tar_data
                )
                update_policy_results = algo.train_rpo(
                    data,
                    advantages,
                    values,
                    tar_data,
                    tar_advantages,
                    tar_values,
                    now_policy_step,
                )
                now_policy_step += 1

        test_results = evaluate(settings)

        logger.record("global/stage", T)
        logger.record("global/env_step", now_real_step)
        logger.record("global/update_step", now_policy_step)
        for key, val in test_results.items():
            logger.record(f"test/{key}", val)
        for key, val in train_real_results.items():
            logger.record(f"train/{key}", val)
        for key, val in train_virt_results.items():
            logger.record(f"train/{key}", val)
        for key, val in update_policy_results.items():
            logger.record(f"update/{key}", val)
        for key, val in update_model_results.items():
            logger.record(f"update/{key}", val)
        logger.dump(T)

        if T % FLAGS.ckpt.n_save_stages == 0:
            np.save(f"{FLAGS.log_dir}/stage_{str(T).zfill(3)}", saver.state_dict())

    np.save(f"{FLAGS.log_dir}/final", saver.state_dict())


if __name__ == "__main__":
    with tf.Session(config=get_tf_config()):
        main()
