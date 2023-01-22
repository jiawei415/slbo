# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import List
import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from slbo.utils.normalizer import Normalizers
from slbo.v_function import BaseVFunction


class RTO(nn.Module):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        vfn: BaseVFunction,
        model: nn.Module,
        normalizers: Normalizers,
        criterion: nn.Module,
        action_type: str,
        step=2,
        lr=1e-3,
        weight_decay=1e-5,
        max_grad_norm=2.0,
        rto_coef=1.0,
        sp_coef=1.0,
        **kwargs,
    ):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.vf = vfn
        self.model = model
        self.criterion = criterion
        self.normalizers = normalizers
        self.step = step
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.rto_coef = rto_coef
        self.sp_coef = sp_coef
        self.action_type = action_type

        with self.scope:
            self.op_states = tf.placeholder(tf.float32, shape=[step, None, dim_state])
            self.op_actions = tf.placeholder(tf.float32, shape=[step, None, dim_action])
            self.op_masks = tf.placeholder(tf.float32, shape=[step, None])
            self.op_next_states = tf.placeholder(
                tf.float32, shape=[step, None, dim_state]
            )

        self.compute_rto_loss()
        self.build_rto_backward()

    def compute_rto_loss(self):
        rto_losses, dy_losses = [], []
        cur_states = self.op_states[0]
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
        self.op_dy_loss = dy_loss
        self.op_rto_loss = rto_loss

    def build_rto_backward(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        params = self.model.parameters()
        regularization = self.weight_decay * tf.add_n(
            [tf.nn.l2_loss(t) for t in params], name="regularization"
        )

        loss = self.rto_coef * self.op_rto_loss + self.sp_coef * (
            self.op_dy_loss + regularization
        )
        grads_and_vars = optimizer.compute_gradients(loss, var_list=params)
        clip_grads, self.op_grad_norm = tf.clip_by_global_norm(
            [grad for grad, _ in grads_and_vars], self.max_grad_norm
        )
        clip_grads_and_vars = [
            (grad, var) for grad, (_, var) in zip(clip_grads, grads_and_vars)
        ]
        self.op_train = optimizer.apply_gradients(clip_grads_and_vars)

    @nn.make_method(fetch="rto_loss dy_loss")
    def get_rto_loss(self, states, next_states, actions, masks) -> List[np.ndarray]:
        pass

    def train(self, states, next_states, actions, masks):
        if self.action_type == "discrete":
            actions = actions.astype(np.int64)
            one_hot_actions = np.zeros(actions.shape + (self.dim_action,))
            for i in range(actions.shape[0]):
                one_hot_actions[i, np.arange(actions.shape[1]), actions[0]] = 1
            actions = one_hot_actions
        rto_loss, dy_loss, grad_norm, _ = self.get_rto_loss(
            states,
            next_states,
            actions,
            masks,
            fetch="rto_loss dy_loss grad_norm train",
        )
        for param in self.model.parameters():
            param.invalidate()
        results = {
            "rto_loss": rto_loss,
            "dy_loss": dy_loss,
            "grad_norm": grad_norm,
        }
        return results
