# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import List, Callable
import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from lunzi import Tensor
from lunzi.Logger import logger
from slbo.utils.dataset import Dataset
from slbo.policies import BaseNNPolicy
from slbo.v_function import BaseVFunction


class PPO(nn.Module):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        policy: BaseNNPolicy,
        vfn: BaseVFunction,
        n_update: int,
        vf_coef=0.25,
        ent_coef=0.0,
        lr=3e-4,
        lr_decay=True,
        clip_range=0.2,
        max_grad_norm=0.5,
        batch_size=64,
        n_opt_epochs=10,
    ):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.policy = policy
        self.vf = vfn
        self.lr = lr
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.clip_range = clip_range
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_opt_epochs = n_opt_epochs
        self.n_update = n_update
        self.lr_decay = lr_decay

        self.old_policy: nn.Module = policy.clone()

        with self.scope:
            self.op_returns = tf.placeholder(
                dtype=tf.float32, shape=[None], name="returns"
            )
            self.op_advantages = tf.placeholder(
                dtype=tf.float32, shape=[None], name="advantages"
            )
            self.op_oldvalues = tf.placeholder(
                dtype=tf.float32, shape=[None], name="oldvalue"
            )
            self.op_states = tf.placeholder(
                dtype=tf.float32, shape=[None, dim_state], name="states"
            )
            self.op_actions = tf.placeholder(
                dtype=tf.float32, shape=[None, dim_action], name="actions"
            )
            self.op_lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

        self.op_sync_old = self.update_old_policy()

        (
            self.op_loss,
            self.op_pg_loss,
            self.op_vf_loss,
            self.op_grad_norm,
            self.op_train,
        ) = self.compute_loss(
            self.op_states,
            self.op_actions,
            self.op_advantages,
            self.op_returns,
            self.op_oldvalues,
            self.op_lr,
        )

    def update_old_policy(self):
        params = self.policy.parameters()
        old_params = self.old_policy.parameters()
        sync_old = tf.group(
            *[tf.assign(old_v, new_v) for old_v, new_v in zip(old_params, params)]
        )
        return sync_old

    def compute_loss(self, states, actions, advantages, returns, oldvalues, lr):
        old_distribution: tf.distributions.Normal = self.old_policy(states)
        distribution: tf.distributions.Normal = self.policy(states)

        # entropy loss
        entropy = distribution.entropy().reduce_sum(axis=1).reduce_mean()

        # policy loss
        ratios: Tensor = (
            (distribution.log_prob(actions) - old_distribution.log_prob(actions))
            .reduce_sum(axis=1)
            .exp()
        )
        pg_losses1 = advantages * ratios
        pg_losses2 = advantages * tf.clip_by_value(
            ratios, 1.0 - self.clip_range, 1.0 + self.clip_range
        )
        pg_loss = -tf.minimum(pg_losses1, pg_losses2).reduce_mean()

        # value loss
        vpred = self.vf(states)
        vpredclipped = oldvalues + tf.clip_by_value(
            vpred - oldvalues, -self.clip_range, self.clip_range
        )
        vf_losses1 = nn.MSELoss()(vpred, returns)
        vf_losses2 = nn.MSELoss()(vpredclipped, returns)
        vf_loss = tf.maximum(vf_losses1, vf_losses2).reduce_mean()

        loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

        optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-5)
        # train_op = optimizer.minimize(loss)
        params = self.policy.parameters() + self.vf.parameters()
        grads = tf.gradients(loss, params)
        if self.max_grad_norm > 0:
            grads, grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
            grads_params = list(zip(grads, params))
        else:
            grad_norm = tf.global_norm(grads)
            grads_params = list(zip(grads, params))
        train_op = optimizer.apply_gradients(grads_params)

        return loss, pg_loss, vf_loss, grad_norm, train_op

    @nn.make_method(fetch="sync_old")
    def sync_old(self) -> List[np.ndarray]:
        pass

    @nn.make_method(fetch="loss")
    def get_loss(
        self, states, actions, advantages, returns, oldvalues, lr
    ) -> List[np.ndarray]:
        pass

    def train(self, samples, advantages, values, update, normalizer=None):
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / np.maximum(advantages.std(), 1e-8)
        assert np.isfinite(advantages).all()

        self.sync_old()
        dataset = Dataset.fromarrays(
            [samples.state, samples.action, advantages, returns, values],
            dtype=[
                ("state", ("f8", self.dim_state)),
                ("action", ("f8", self.dim_action)),
                ("advantages", "f8"),
                ("returns", "f8"),
                ("oldvalues", "f8"),
            ],
        )
        frac = 1.0 - update / self.n_update
        lr = self.lr * frac if self.lr_decay else self.lr
        losses, pg_losses, vf_losses, grads_norm = [], [], [], []
        for _ in range(self.n_opt_epochs):
            for subset in dataset.iterator(self.batch_size):
                loss, pg_loss, vf_loss, grad_norm, now_lr, _ = self.get_loss(
                    subset.state,
                    subset.action,
                    subset.advantages,
                    subset.returns,
                    subset.oldvalues,
                    lr,
                    fetch="loss pg_loss vf_loss grad_norm lr train",
                )
                losses.append(loss)
                pg_losses.append(pg_loss)
                vf_losses.append(vf_loss)
                grads_norm.append(grad_norm)

        for param in self.parameters():
            param.invalidate()

        assert np.isclose(lr, now_lr)

        return np.mean(losses), np.mean(pg_losses), np.mean(vf_losses), np.mean(grads_norm), now_lr
