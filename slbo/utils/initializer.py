import numpy as np
import tensorflow as tf


# copied from https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L97-L102
def normc_initializer(std=1.0, axis=0):
    def _initializer(shape, dtype=tf.float32, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return tf.constant(out)
    return _initializer
