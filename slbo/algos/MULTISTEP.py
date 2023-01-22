# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from lunzi import Tensor
from slbo.utils.normalizer import Normalizers


class MultiStep(nn.Module):
    op_train: Tensor
    op_grad_norm: Tensor
    _step: int
    _criterion: nn.Module
    _normalizers: Normalizers
    _model: nn.Module

    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        model: nn.Module,
        normalizers: Normalizers,
        criterion: nn.Module,
        action_type: str,
        step=2,
        lr=1e-3,
        weight_decay=1e-5,
        max_grad_norm=2.0,
    ):
        super().__init__()
        self._step = step
        self._criterion = criterion
        self._model = model
        self._normalizers = normalizers
        self.step = step
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.action_type = action_type
        with self.scope:
            self.op_states = tf.placeholder(tf.float32, shape=[step, None, dim_state])
            self.op_actions = tf.placeholder(tf.float32, shape=[step, None, dim_action])
            self.op_masks = tf.placeholder(tf.float32, shape=[step, None])
            self.op_next_states_ = tf.placeholder(
                tf.float32, shape=[step, None, dim_state]
            )

        self.compute_loss()
        self.build_backward()

    def compute_loss(self):
        """
        All inputs have shape [num_steps, batch_size, xxx]
        """
        losses = []
        cur_states = self.op_states[0]
        for i in range(self._step):
            next_states = self._model(cur_states, self.op_actions[i])
            diffs = (
                next_states - cur_states - self.op_next_states_[i] + self.op_states[i]
            )
            weighted_diffs = diffs / self._normalizers.diff.op_std.maximum(1e-6)
            losses.append(self._criterion(weighted_diffs, 0, cur_states))

            if i < self._step - 1:
                cur_states = self.op_states[i + 1] + self.op_masks[i].expand_dims(
                    -1
                ) * (next_states - self.op_states[i + 1])

        loss = tf.reduce_mean(tf.add_n(losses) / self.step)
        self.op_loss = loss

    @nn.make_method(fetch="loss")
    def get_loss(self, states, next_states_, actions, masks):
        pass

    def build_backward(self):
        loss = self.op_loss.reduce_mean(name="Loss")

        optimizer = tf.train.AdamOptimizer(self.lr)
        params = self._model.parameters()
        regularization = self.weight_decay * tf.add_n(
            [tf.nn.l2_loss(t) for t in params], name="regularization"
        )

        grads_and_vars = optimizer.compute_gradients(
            loss + regularization, var_list=params
        )
        print([var.name for grad, var in grads_and_vars])
        clip_grads, self.op_grad_norm = tf.clip_by_global_norm(
            [grad for grad, _ in grads_and_vars], self.max_grad_norm
        )
        clip_grads_and_vars = [
            (grad, var) for grad, (_, var) in zip(clip_grads, grads_and_vars)
        ]
        self.op_train = optimizer.apply_gradients(clip_grads_and_vars)

    def train(self, states, next_states, actions, masks):
        if self.action_type == "discrete":
            actions = actions.astype(np.int64)
            one_hot_actions = np.zeros(actions.shape + (self.dim_action,))
            for i in range(actions.shape[0]):
                one_hot_actions[i, np.arange(actions.shape[1]), actions[0]] = 1
            actions = one_hot_actions
        loss, grad_norm, _ = self.get_loss(
            states,
            next_states,
            actions,
            masks,
            fetch="loss grad_norm train",
        )
        for param in self._model.parameters():
            param.invalidate()

        results = {
            "loss": loss,
            "grad_norm": grad_norm,
        }
        return results
