# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import List, Callable
import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from lunzi import Tensor
from slbo.utils.dataset import Dataset
from slbo.utils.normalizer import Normalizers
from slbo.policies import BaseNNPolicy
from slbo.v_function import BaseVFunction


class RPO(nn.Module):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        policy: BaseNNPolicy,
        vfn: BaseVFunction,
        n_update: int,
        action_type: str,
        vf_coef=0.25,
        ent_coef=0.0,
        sim_coef=1.0,
        tar_coef=1.0,
        lr=1e-3,
        lr_min=3e-4,
        lr_decay=True,
        clip_range=0.2,
        max_grad_norm=0.5,
        batch_size=64,
        n_opt_epochs=10,
        norm_sim_adv=True,
        norm_tar_adv=True,
        **kwargs,
    ):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.policy = policy
        self.vf = vfn
        self.lr = lr
        self.lr_min = lr_min
        self.lr_decay = lr_decay
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.sim_coef = sim_coef
        self.tar_coef = tar_coef
        self.clip_range = clip_range
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_opt_epochs = n_opt_epochs
        self.norm_sim_adv = norm_sim_adv
        self.norm_tar_adv = norm_tar_adv
        self.n_update = n_update
        self.action_type = action_type

        self.old_policy: nn.Module = policy.clone()

        with self.scope:
            # for rpo
            self.op_sim_returns = tf.placeholder(
                dtype=tf.float32, shape=[None], name="sim_returns"
            )
            self.op_sim_advantages = tf.placeholder(
                dtype=tf.float32, shape=[None], name="sim_advantages"
            )
            self.op_sim_oldvalues = tf.placeholder(
                dtype=tf.float32, shape=[None], name="sim_oldvalues"
            )
            self.op_sim_states = tf.placeholder(
                dtype=tf.float32, shape=[None, dim_state], name="sim_states"
            )
            self.op_tar_returns = tf.placeholder(
                dtype=tf.float32, shape=[None], name="tar_returns"
            )
            self.op_tar_advantages = tf.placeholder(
                dtype=tf.float32, shape=[None], name="tar_advantages"
            )
            self.op_tar_oldvalues = tf.placeholder(
                dtype=tf.float32, shape=[None], name="tar_oldvalues"
            )
            self.op_tar_states = tf.placeholder(
                dtype=tf.float32, shape=[None, dim_state], name="tar_states"
            )

            self._dtype = [
                ("sim_state", ("f8", self.dim_state)),
                ("sim_advantages", "f8"),
                ("sim_returns", "f8"),
                ("sim_oldvalues", "f8"),
                ("tar_state", ("f8", self.dim_state)),
                ("tar_advantages", "f8"),
                ("tar_returns", "f8"),
                ("tar_oldvalues", "f8"),
            ]

            if action_type == "continuous":
                self.op_tar_actions = tf.placeholder(
                    dtype=tf.float32, shape=[None, dim_action], name="tar_ractions"
                )
                self.op_sim_actions = tf.placeholder(
                    dtype=tf.float32, shape=[None, dim_action], name="sim_actions"
                )
                self._dtype.append(("sim_action", "f8", (self.dim_action,)))
                self._dtype.append(("tar_action", "f8", (self.dim_action,)))
            elif action_type == "discrete":
                self.op_tar_actions = tf.placeholder(
                    dtype=tf.float32, shape=[None], name="tar_ractions"
                )
                self.op_sim_actions = tf.placeholder(
                    dtype=tf.float32, shape=[None], name="sim_actions"
                )
                self._dtype.append(("sim_action", "f8"))
                self._dtype.append(("tar_action", "f8"))
            self.op_lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

        self.update_old_policy()
        self.compute_rpo_loss()
        self.build_rpo_backward()

    def update_old_policy(self):
        params = self.policy.parameters()
        old_params = self.old_policy.parameters()
        sync_old = tf.group(
            *[tf.assign(old_v, new_v) for old_v, new_v in zip(old_params, params)]
        )
        self.op_sync_old = sync_old

    def compute_rpo_loss(self):
        # value loss
        vpred = self.vf(self.op_sim_states)
        vpredclipped = self.op_sim_oldvalues + tf.clip_by_value(
            vpred - self.op_sim_oldvalues, -self.clip_range, self.clip_range
        )
        vf_losses1 = nn.MSELoss()(vpred, self.op_sim_returns)
        vf_losses2 = nn.MSELoss()(vpredclipped, self.op_sim_returns)
        vf_loss = tf.maximum(vf_losses1, vf_losses2).reduce_mean()
        self.op_vf_loss = vf_loss

        # tar policy loss
        tar_distribution_old = self.old_policy(self.op_tar_states)
        tar_distribution = self.policy(self.op_tar_states)

        tar_ratios = tar_distribution.log_prob(
            self.op_tar_actions
        ) - tar_distribution_old.log_prob(self.op_tar_actions)
        if len(tar_ratios.shape) > 1:
            tar_ratios = tar_ratios.reduce_mean(axis=1)
        tar_ratios = tar_ratios.exp()

        pg_tar_losses1 = self.op_tar_advantages * tar_ratios
        pg_tar_losses2 = self.op_tar_advantages * tf.clip_by_value(
            tar_ratios, 1.0 - self.clip_range, 1.0 + self.clip_range
        )
        pg_tar_loss = -tf.minimum(pg_tar_losses1, pg_tar_losses2).reduce_mean()
        self.op_pg_tar_loss = pg_tar_loss

        # sim policy loss
        sim_distribution_old = self.old_policy(self.op_sim_states)
        sim_distribution = self.policy(self.op_sim_states)

        sim_ratios = sim_distribution.log_prob(
            self.op_sim_actions
        ) - sim_distribution_old.log_prob(self.op_sim_actions)
        if len(sim_ratios.shape) > 1:
            sim_ratios = sim_ratios.reduce_mean(axis=1)
        sim_ratios = sim_ratios.exp()

        pg_sim_losses1 = self.op_sim_advantages * sim_ratios
        pg_sim_losses2 = self.op_sim_advantages * tf.clip_by_value(
            sim_ratios, 1.0 - self.clip_range, 1.0 + self.clip_range
        )
        pg_sim_loss = -tf.minimum(pg_sim_losses1, pg_sim_losses2).reduce_mean()
        self.op_pg_sim_loss = pg_sim_loss

        # entropy loss
        entropy = sim_distribution.entropy()
        if len(entropy.shape) > 1:
            entropy = entropy.reduce_sum(axis=1)
        entropy = tf.reduce_mean(entropy)
        self.op_entropy = entropy

        rpo_loss = (
            pg_sim_loss * self.sim_coef
            - entropy * self.ent_coef
            + vf_loss * self.vf_coef
            + pg_tar_loss * self.tar_coef
        )
        self.op_rpo_loss = rpo_loss

    def build_rpo_backward(self):
        optimizer = tf.train.AdamOptimizer(self.op_lr)
        params = self.policy.parameters() + self.vf.parameters()
        grads = tf.gradients(self.op_rpo_loss, params)
        if self.max_grad_norm > 0:
            grads, self.op_grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
            grads_params = list(zip(grads, params))
        else:
            self.op_grad_norm = tf.global_norm(grads)
            grads_params = list(zip(grads, params))
        self.op_train = optimizer.apply_gradients(grads_params)

    @nn.make_method(fetch="sync_old")
    def sync_old(self) -> List[np.ndarray]:
        pass

    @nn.make_method(fetch="rpo_loss")
    def get_rpo_loss(
        self,
        sim_states,
        sim_actions,
        sim_advantages,
        sim_returns,
        sim_oldvalues,
        tar_states,
        tar_actions,
        tar_advantages,
        tar_returns,
        tar_oldvalues,
        lr,
    ) -> List[np.ndarray]:
        pass

    def train(
        self,
        sim_samples,
        sim_advantages,
        sim_values,
        tar_samples,
        tar_advantages,
        tar_values,
        update,
    ):
        sim_returns = sim_advantages + sim_values
        if self.norm_sim_adv:
            sim_advantages = (sim_advantages - sim_advantages.mean()) / np.maximum(
                sim_advantages.std(), 1e-8
            )
        tar_returns = tar_advantages + tar_values
        if self.norm_tar_adv:
            tar_advantages = (tar_advantages - tar_advantages.mean()) / np.maximum(
                tar_advantages.std(), 1e-8
            )
        assert np.isfinite(sim_advantages).all() and np.isfinite(tar_advantages).all()

        self.sync_old()
        dataset = Dataset.fromarrays(
            [
                sim_samples.state,
                sim_advantages,
                sim_returns,
                sim_values,
                tar_samples.state,
                tar_advantages,
                tar_returns,
                tar_values,
                sim_samples.action,
                tar_samples.action,
            ],
            dtype=self._dtype,
        )
        frac = 1.0 - update / self.n_update
        lr = max(self.lr * frac, self.lr_min) if self.lr_decay else self.lr
        rpo_losses, pg_sim_losses, pg_tar_losses, vf_losses, grads_norm = (
            [],
            [],
            [],
            [],
            [],
        )
        for _ in range(self.n_opt_epochs):
            for subset in dataset.iterator(self.batch_size):
                (
                    rpo_loss,
                    pg_sim_loss,
                    pg_tar_loss,
                    vf_loss,
                    grad_norm,
                    now_lr,
                    _,
                ) = self.get_rpo_loss(
                    subset.sim_state,
                    subset.sim_action,
                    subset.sim_advantages,
                    subset.sim_returns,
                    subset.sim_oldvalues,
                    subset.tar_state,
                    subset.tar_action,
                    subset.tar_advantages,
                    subset.tar_returns,
                    subset.tar_oldvalues,
                    lr,
                    fetch="rpo_loss pg_sim_loss pg_tar_loss vf_loss grad_norm lr train",
                )
                rpo_losses.append(rpo_loss)
                pg_sim_losses.append(pg_sim_loss)
                pg_tar_losses.append(pg_tar_loss)
                vf_losses.append(vf_loss)
                grads_norm.append(grad_norm)

        for param in self.parameters():
            param.invalidate()

        assert np.isclose(lr, now_lr)

        results = {
            "rpo_loss": np.mean(rpo_losses),
            "pg_sim_loss": np.mean(pg_sim_loss),
            "pg_tar_loss": np.mean(pg_tar_loss),
            "vf_loss": np.mean(vf_losses),
            "grad_norm": np.mean(grads_norm),
            "lr": float(now_lr),
        }
        return results
