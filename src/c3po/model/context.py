from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import scan
from functools import partial

from .model import MLP


def context_factory(context_model: str, context_dim: int, **kwargs):
    if context_model == "simpleRNN":
        return SimpleRNNContext(context_dim=context_dim, **kwargs)
    else:
        raise ValueError(f"Unknown context model: {context_model}")


class BaseContext(nn.RNNCellBase):
    context_dim: int

    def setup(self):
        pass

    def __call__(self, c, z):
        raise NotImplementedError

    def initialize_carry(self, init_key, input_shape):
        raise NotImplementedError

    @property
    def num_feature_axes(self):
        return 1


class SimpleRNNContext(BaseContext):
    widths: Sequence[int]

    def setup(self):
        self.network = MLP(self.widths + (self.context_dim,))
        self.history_matrix = nn.Dense(
            self.context_dim,
            use_bias=False,
            kernel_init=timescale_initializer,
        )

    def __call__(self, c, z):
        z_trans = jnp.tanh(self.network(z))
        c_trans = self.history_matrix(c)
        c_new = z_trans + c_trans
        return c_new, c_new

    def initialize_carry(self, init_key, input_shape):
        return jax.random.normal(init_key, (input_shape[0], self.context_dim)) * 0.3


def timescale_initializer(key, shape, dtype=jnp.float32, scales=[0.9, 0.9, 0.5]):
    qq = np.random.normal(0, 0.1, shape)
    for i in range(shape[0]):
        qq[i, i] = scales[i]
    return jnp.array(qq)
