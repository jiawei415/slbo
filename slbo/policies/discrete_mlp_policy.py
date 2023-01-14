# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import List
import tensorflow as tf
import numpy as np
from lunzi import Tensor
from lunzi import nn
from slbo.utils.initializer import normc_initializer
from slbo.utils.truncated_normal import LimitedEntNormal
from . import BasePolicy
from slbo.utils.normalizer import GaussianNormalizer


def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)

    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    return x


class DiscreteMLPPolicy(nn.Module, BasePolicy):
    op_states: Tensor

    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        hidden_sizes: List[int],
        normalizer: GaussianNormalizer,
        init_std=1.0,
    ):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.hidden_sizes = hidden_sizes
        self.init_std = init_std
        self.normalizer = normalizer
        with self.scope:
            self.op_states = tf.placeholder(
                tf.float32, shape=[None, dim_state], name="states"
            )

            layers = []
            # note that the placeholder has size 105.
            all_sizes = [dim_state, *self.hidden_sizes]
            for i, (in_features, out_features) in enumerate(
                zip(all_sizes[:-1], all_sizes[1:])
            ):
                layers.append(
                    nn.Linear(
                        in_features,
                        out_features,
                        weight_initializer=normc_initializer(1),
                    )
                )
                layers.append(nn.Tanh())
            layers.append(
                nn.Linear(
                    all_sizes[-1],
                    dim_action,
                    weight_initializer=normc_initializer(0.01),
                )
            )
            self.net = nn.Sequential(*layers)

        self.distribution = self(self.op_states)

        self.register_callable("[states] => [actions]", self.fast)

    def forward(self, states):
        if self.normalizer is not None:
            states = self.normalizer(states)
        actions_logits = self.net(states)
        distribution = tf.distributions.Categorical(actions_logits)

        return distribution

    @nn.make_method(fetch="actions")
    def get_actions(self, states):
        pass

    def fast(self, states):
        if self.normalizer is not None:
            states = self.normalizer.fast(states)
        actions_logits = self.net.fast(states)
        actions_prob = softmax(actions_logits)
        actions = []
        for action_prob in actions_prob:
            actions.append(np.random.choice(len(action_prob), p=action_prob))
        return np.array(actions)

    def clone(self):
        return DiscreteMLPPolicy(
            self.dim_state,
            self.dim_action,
            self.hidden_sizes,
            self.normalizer,
        )
