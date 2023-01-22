# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import List
import tensorflow as tf
import numpy as np
import lunzi.nn as nn
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
        action_type: str,
        vf_coef=0.25,
        ent_coef=0.0,
        lr=1e-3,
        lr_min=3e-4,
        lr_decay=True,
        clip_range=0.2,
        max_grad_norm=0.5,
        batch_size=64,
        n_opt_epochs=10,
        norm_adv=True,
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
        self.clip_range = clip_range
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_opt_epochs = n_opt_epochs
        self.n_update = n_update
        self.action_type = action_type
        self.norm_adv = norm_adv

        self.old_policy: nn.Module = policy.clone()

        with self.scope:
            self.op_returns = tf.placeholder(
                dtype=tf.float32, shape=[None], name="returns"
            )
            self.op_advantages = tf.placeholder(
                dtype=tf.float32, shape=[None], name="advantages"
            )
            self.op_oldvalues = tf.placeholder(
                dtype=tf.float32, shape=[None], name="oldvalues"
            )
            self.op_states = tf.placeholder(
                dtype=tf.float32, shape=[None, dim_state], name="states"
            )

            self._dtype = [
                ("advantages", "f8"),
                ("returns", "f8"),
                ("oldvalues", "f8"),
                ("state", ("f8", self.dim_state)),
            ]

            if action_type == "continuous":
                self.op_actions = tf.placeholder(
                    dtype=tf.float32, shape=[None, dim_action], name="actions"
                )
                self._dtype.append(("action", "f8", (self.dim_action,)))
            elif action_type == "discrete":
                self.op_actions = tf.placeholder(
                    dtype=tf.float32, shape=[None], name="actions"
                )
                self._dtype.append(("action", "f8"))
            self.op_lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

        self.update_old_policy()
        self.compute_loss()
        self.build_backward()

    def update_old_policy(self):
        params = self.policy.parameters()
        old_params = self.old_policy.parameters()
        sync_old = tf.group(
            *[tf.assign(old_v, new_v) for old_v, new_v in zip(old_params, params)]
        )
        self.op_sync_old = sync_old

    def compute_loss(self):
        old_distribution = self.old_policy(self.op_states)
        distribution = self.policy(self.op_states)

        # entropy loss
        entropy = distribution.entropy()
        if len(entropy.shape) > 1:
            entropy = entropy.reduce_sum(axis=1)
        entropy = tf.reduce_mean(entropy)
        self.op_entropy = entropy

        # policy loss
        ratios = distribution.log_prob(self.op_actions) - old_distribution.log_prob(
            self.op_actions
        )
        if len(ratios.shape) > 1:
            ratios = ratios.reduce_mean(axis=1)
        ratios = ratios.exp()

        pg_losses1 = self.op_advantages * ratios
        pg_losses2 = self.op_advantages * tf.clip_by_value(
            ratios, 1.0 - self.clip_range, 1.0 + self.clip_range
        )
        pg_loss = -tf.minimum(pg_losses1, pg_losses2).reduce_mean()
        self.op_pg_loss = pg_loss

        # value loss
        vpred = self.vf(self.op_states)
        vpredclipped = self.op_oldvalues + tf.clip_by_value(
            vpred - self.op_oldvalues, -self.clip_range, self.clip_range
        )
        vf_losses1 = nn.MSELoss()(vpred, self.op_returns)
        vf_losses2 = nn.MSELoss()(vpredclipped, self.op_returns)
        vf_loss = tf.maximum(vf_losses1, vf_losses2).reduce_mean()
        self.op_vf_loss = vf_loss

        loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef
        self.op_loss = loss

    def build_backward(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.op_lr, epsilon=1e-5)
        params = self.policy.parameters() + self.vf.parameters()
        grads = tf.gradients(self.op_loss, params)
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

    @nn.make_method(fetch="loss")
    def get_loss(
        self, states, actions, advantages, returns, oldvalues, lr
    ) -> List[np.ndarray]:
        pass

    def train(self, samples, advantages, values, update):
        returns = advantages + values
        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / np.maximum(
                advantages.std(), 1e-8
            )
        assert np.isfinite(advantages).all()

        self.sync_old()
        dataset = Dataset.fromarrays(
            [advantages, returns, values, samples.state, samples.action],
            dtype=self._dtype,
        )
        frac = 1.0 - update / self.n_update
        lr = max(self.lr * frac, self.lr_min) if self.lr_decay else self.lr
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

        results = {
            "loss": np.mean(losses),
            "pg_loss": np.mean(pg_loss),
            "vf_loss": np.mean(vf_losses),
            "grad_norm": np.mean(grads_norm),
            "lr": float(now_lr),
        }
        return results
