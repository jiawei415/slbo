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


class RPTO(nn.Module):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        policy: BaseNNPolicy,
        vfn: BaseVFunction,
        model: nn.Module,
        normalizers: Normalizers,
        criterion: nn.Module,
        n_update: int,
        action_type: str,
        # model
        step=4,
        weight_decay=1e-5,
        model_max_grad_norm=2.0,
        model_lr=1e-3,
        rto_coef=1.0,
        sp_coef=1.0,
        # policy
        vf_coef=0.25,
        ent_coef=0.0,
        rpo_coef=1.0,
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
        self.model = model
        self.criterion = criterion
        self.normalizers = normalizers
        self.step = step
        self.weight_decay = weight_decay
        self.model_max_grad_norm = model_max_grad_norm
        self.model_lr = model_lr
        self.rto_coef = rto_coef
        self.sp_coef = sp_coef
        self.lr = lr
        self.lr_min = lr_min
        self.lr_decay = lr_decay
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.rpo_coef = rpo_coef
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
            if action_type == "continuous":
                self.op_tar_actions = tf.placeholder(
                    dtype=tf.float32, shape=[None, dim_action], name="tar_ractions"
                )
                self.op_sim_actions = tf.placeholder(
                    dtype=tf.float32, shape=[None, dim_action], name="sim_actions"
                )
            elif action_type == "discrete":
                self.op_tar_actions = tf.placeholder(
                    dtype=tf.float32, shape=[None], name="tar_ractions"
                )
                self.op_sim_actions = tf.placeholder(
                    dtype=tf.float32, shape=[None], name="sim_actions"
                )
            # for rto
            self.op_states = tf.placeholder(tf.float32, shape=[step, None, dim_state])
            self.op_actions = tf.placeholder(tf.float32, shape=[step, None, dim_action])
            self.op_masks = tf.placeholder(tf.float32, shape=[step, None])
            self.op_next_states = tf.placeholder(
                tf.float32, shape=[step, None, dim_state]
            )
            self.op_lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

        self.op_sync_old = self.update_old_policy()

        (
            self.op_rpo_loss,
            self.op_pg_sim_loss,
            self.op_pg_tar_loss,
            self.op_vf_loss,
            self.op_entropy,
        ) = self.compute_rpo_loss()
        self.op_dy_loss, self.op_rto_loss = self.compute_rto_loss()
        self.build_rpo_backward()
        self.build_rto_backward()

    def update_old_policy(self):
        params = self.policy.parameters()
        old_params = self.old_policy.parameters()
        sync_old = tf.group(
            *[tf.assign(old_v, new_v) for old_v, new_v in zip(old_params, params)]
        )
        return sync_old

    def compute_rpo_loss(self):
        # value loss
        vpred = self.vf(self.op_sim_states)
        vpredclipped = self.op_sim_oldvalues + tf.clip_by_value(
            vpred - self.op_sim_oldvalues, -self.clip_range, self.clip_range
        )
        vf_losses1 = nn.MSELoss()(vpred, self.op_sim_returns)
        vf_losses2 = nn.MSELoss()(vpredclipped, self.op_sim_returns)
        vf_loss = tf.maximum(vf_losses1, vf_losses2).reduce_mean()

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

        # entropy loss
        entropy = sim_distribution.entropy()
        if len(entropy.shape) > 1:
            entropy = entropy.reduce_sum(axis=1)
        entropy = tf.reduce_mean(entropy)

        rpo_loss = (
            pg_sim_loss * self.rpo_coef
            - entropy * self.ent_coef
            + vf_loss * self.vf_coef
            + pg_tar_loss * self.tar_coef
        )

        return rpo_loss, pg_sim_loss, pg_tar_loss, vf_loss, entropy

    def compute_rto_loss(self):

        cur_states = self.op_states[0]
        rto_losses, dy_losses = [], []
        for i in range(self.step):
            next_states = self.model(cur_states, self.op_actions[i])
            diffs = (
                next_states - cur_states - self.op_next_states[i] + self.op_states[i]
            )
            weighted_diffs = diffs / self.normalizers.diff.op_std.maximum(1e-6)
            dy_losses.append(self.criterion(weighted_diffs, 0, cur_states))
            rto_losses.append(
                nn.MSELoss()(
                    self.vf(next_states),
                    self.vf(self.op_next_states[i]) * self.op_masks[i].expand_dims(-1),
                )
            )

            if i < self.step - 1:
                cur_states = self.op_states[i + 1] + self.op_masks[i].expand_dims(
                    -1
                ) * (next_states - self.op_states[i + 1])
        dy_loss = tf.reduce_mean(tf.add_n(dy_losses) / self.step)
        rto_loss = tf.reduce_mean(tf.add_n(rto_losses) / self.step)

        return dy_loss, rto_loss

    def build_rpo_backward(self):
        optimizer = tf.train.AdamOptimizer(self.op_lr)
        params = self.policy.parameters() + self.vf.parameters()
        grads = tf.gradients(self.op_rpo_loss, params)
        if self.max_grad_norm > 0:
            grads, self.op_policy_grad_norm = tf.clip_by_global_norm(
                grads, self.max_grad_norm
            )
            grads_params = list(zip(grads, params))
        else:
            self.op_policy_grad_norm = tf.global_norm(grads)
            grads_params = list(zip(grads, params))
        self.op_train_policy = optimizer.apply_gradients(grads_params)

    def build_rto_backward(self):

        optimizer = tf.train.AdamOptimizer(self.model_lr)
        params = self.model.parameters()
        regularization = self.weight_decay * tf.add_n(
            [tf.nn.l2_loss(t) for t in params], name="regularization"
        )

        loss = self.rto_coef * self.op_rto_loss + self.sp_coef * (
            self.op_dy_loss + regularization
        )
        grads_and_vars = optimizer.compute_gradients(loss, var_list=params)
        clip_grads, self.op_model_grad_norm = tf.clip_by_global_norm(
            [grad for grad, _ in grads_and_vars], self.model_max_grad_norm
        )
        clip_grads_and_vars = [
            (grad, var) for grad, (_, var) in zip(clip_grads, grads_and_vars)
        ]
        self.op_train_model = optimizer.apply_gradients(clip_grads_and_vars)

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

    @nn.make_method(fetch="rto_loss dy_loss")
    def get_rto_loss(self, states, next_states, actions, masks) -> List[np.ndarray]:
        pass

    def train_rto(self, states, next_states, actions, masks):
        if self.action_type == "discrete":
            one_hot_actions = np.zeros(actions.shape + (self.dim_action,))
            for i in range(actions.shape[0]):
                one_hot_actions[
                    i, np.arange(actions.shape[1]), actions[0].astype(np.int64)
                ] = 1
            actions = one_hot_actions
        rto_loss, dy_loss, model_grad_norm, _ = self.get_rto_loss(
            states,
            next_states,
            actions,
            masks,
            fetch="rto_loss dy_loss model_grad_norm train_model",
        )
        for param in self.model.parameters():
            param.invalidate()
        return rto_loss, dy_loss, model_grad_norm

    def train_rpo(
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
        dtype = [
            ("sim_state", ("f8", self.dim_state)),
            ("sim_advantages", "f8"),
            ("sim_returns", "f8"),
            ("sim_oldvalues", "f8"),
            ("tar_state", ("f8", self.dim_state)),
            ("tar_advantages", "f8"),
            ("tar_returns", "f8"),
            ("tar_oldvalues", "f8"),
        ]
        if self.action_type == "continuous":
            dtype.append(("sim_action", "f8", (self.dim_action,)))
            dtype.append(("tar_action", "f8", (self.dim_action,)))
        elif self.action_type == "discrete":
            dtype.append(("sim_action", "f8"))
            dtype.append(("tar_action", "f8"))
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
            dtype=dtype,
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
                    fetch="rpo_loss pg_sim_loss pg_tar_loss vf_loss policy_grad_norm lr train_policy",
                )
                rpo_losses.append(rpo_loss)
                pg_sim_losses.append(pg_sim_loss)
                pg_tar_losses.append(pg_tar_loss)
                vf_losses.append(vf_loss)
                grads_norm.append(grad_norm)

        for param in self.policy.parameters():
            param.invalidate()
        for param in self.old_policy.parameters():
            param.invalidate()
        for param in self.vf.parameters():
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
