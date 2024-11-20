from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import scan
from functools import partial

from .util import MLP, StabilizedDenseDynamics


def context_factory(context_model: str, context_dim: int, **kwargs):
    if context_model == "simpleRNN":
        return SimpleRNNContext(context_dim=context_dim, **kwargs)
    elif context_model == "stableSimpleRNN":
        return StableSimpleRNNContext(context_dim=context_dim, **kwargs)
    elif context_model == "fastSlowRNN":
        return FastSlowRNNContext(context_dim=context_dim, **kwargs)
    elif context_model == "fastSlowRNN_v2":
        return FastSlowRNNContext_v2(context_dim=context_dim, **kwargs)
    elif context_model == "lfads":
        return LFADSContext(context_dim=context_dim, **kwargs)
    elif context_model == "LSTM":
        return nn.LSTMCell(name="context", features=context_dim, **kwargs)
    elif context_model == "GRU":
        return nn.GRUCell(name="context", features=context_dim, **kwargs)
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
    scales: Sequence[float] = None

    def setup(self):
        self.network = MLP(self.widths + (self.context_dim,))
        self.history_matrix = nn.Dense(
            self.context_dim,
            use_bias=False,
            kernel_init=partial(timescale_initializer, scales=self.scales),
        )

    def __call__(self, c, z):
        z_trans = jnp.tanh(self.network(z))
        c_trans = self.history_matrix(c)
        c_new = z_trans + c_trans
        # c_new = jnp.clip(c_new, -30, 30)
        return c_new, c_new

    def initialize_carry(self, init_key, input_shape):
        return jax.random.normal(init_key, (input_shape[0], self.context_dim)) * 0.3


def timescale_initializer(key, shape, dtype=jnp.float32, scales=[0.9, 0.9, 0.5]):
    if scales is None:
        scales = [0.9, 0.9, 0.5]
    # qq = np.random.normal(0, 0.1, shape)
    qq = np.random.normal(0, 0.01, shape)
    for i in range(shape[0]):
        qq[i, i] = scales[i]
    return jnp.array(qq)


class LFADSContext(BaseContext):
    widths: Sequence[int]

    def setup(self):
        self.network = MLP(self.widths + (self.context_dim,))

    def __call__(self, c, delta_t):
        dc_dt = jnp.tanh(self.network(c))
        c_new = c + dc_dt * delta_t
        return c_new, c_new


class StableSimpleRNNContext(BaseContext):
    widths: Sequence[int]
    scales: Sequence[float] = None

    def setup(self):
        self.network = MLP(self.widths + (self.context_dim,))
        self.history_matrix = nn.Dense(
            self.context_dim,
            use_bias=False,
            kernel_init=partial(stable_timescale_initializer, scales=self.scales),
        )

    def __call__(self, c, z):
        z_trans = jnp.tanh(self.network(z))
        c_trans = c - self.history_matrix(c)
        c_new = z_trans + c_trans
        # c_new = jnp.clip(c_new, -30, 30)
        return c_new, c_new

    def initialize_carry(self, init_key, input_shape):
        return jax.random.normal(init_key, (input_shape[0], self.context_dim)) * 0.3


def stable_timescale_initializer(key, shape, dtype=jnp.float32, scales=[0.1, 0.1, 0.1]):
    # if scales is None:
    #     scales = [0.9, 0.9, 0.5]
    # qq = np.random.normal(0, 0.1, shape)
    print(scales, shape)
    qq = np.random.normal(0, 0.1, shape)
    print(qq.shape)
    for i in range(shape[0]):
        qq[i, i] = scales[i]
    return jnp.array(qq)


class FastSlowRNNContext(BaseContext):
    widths: Sequence[int]
    scales: Sequence[float] = None
    latents: int = None
    slow_update_freq: int = 100  # Frequency for updating slow state
    n_slow: int = 2  # Number of slow states
    stabilize_dynamics: bool = True

    def setup(self):
        if self.latents is None:
            self.latents = self.context_dim
        n_fast = self.latents - self.n_slow
        # self.network = MLP(self.widths + (self.context_dim - self.n_slow,))
        self.network = nn.Dense(
            n_fast,
            use_bias=True,
        )

        if self.stabilize_dynamics:
            self.history_matrix_fast = StabilizedDenseDynamics(
                n_fast,
                delta_matrix=False,
                kernel_init=partial(timescale_initializer, scales=self.scales[:n_fast]),
            )
            self.history_matrix_slow = StabilizedDenseDynamics(
                self.n_slow,
                delta_matrix=False,
                kernel_init=partial(timescale_initializer, scales=self.scales[n_fast:]),
            )
        else:
            self.history_matrix_fast = nn.Dense(
                n_fast,
                use_bias=False,
                kernel_init=partial(
                    stable_timescale_initializer, scales=self.scales[:n_fast]
                ),
            )
            self.history_matrix_slow = nn.Dense(
                self.n_slow,
                use_bias=False,
                kernel_init=partial(
                    stable_timescale_initializer, scales=self.scales[n_fast:]
                ),
            )
        self.fast_to_slow = nn.Dense(
            self.n_slow,
            use_bias=False,
            # kernel_init=partial(
            #     stable_timescale_initializer, scales=self.scales[n_fast:]
            # ),
        )

        if self.latents != self.context_dim:
            self.context_projection = nn.Dense(
                self.context_dim,
                use_bias=False,
            )

        self.initial_carry = self.param(
            "initial_carry",  # Parameter name
            lambda key: jax.random.normal(key, (self.latents, 1)) * 0.3,
        )

    def __call__(self, carry, z):
        c_combined, step = carry
        n_fast = self.latents - self.n_slow
        print(step.shape, step.dtype)

        c_fast = c_combined[:, :n_fast]
        c_slow = c_combined[:, n_fast:]
        print(c_combined.shape, c_fast.shape, c_slow.shape)
        # Transform the input z
        z_trans = jnp.tanh(self.network(z))

        # Fast state update
        if self.stabilize_dynamics:
            c_fast_trans = self.history_matrix_fast(c_fast)
        else:
            c_fast_trans = c_fast - self.history_matrix_fast(c_fast)
        c_fast_new = z_trans + c_fast_trans

        # Slow state update only at specified intervals
        update_slow = nn.relu(step - self.slow_update_freq + 1)
        c_fast_slow = self.fast_to_slow(c_fast)
        if self.stabilize_dynamics:
            c_slow_trans = self.history_matrix_slow(c_slow)
            c_slow_trans = c_slow_trans + c_fast_slow
            c_slow_new = c_slow * (1 - update_slow) + c_slow_trans * update_slow
        else:
            c_slow_trans = self.history_matrix_slow(c_slow)
            c_slow_update = c_fast_slow - c_slow_trans
            c_slow_new = c_slow + update_slow * (c_slow_update)
        step = (step) * (1 - update_slow) + 1

        # Combine fast and slow states
        c_combined = jnp.concatenate([c_fast_new, c_slow_new], axis=-1)
        c_combined = jnp.clip(c_combined, -1000, 1000)
        if self.latents != self.context_dim:
            c_proj = self.context_projection(c_combined)
            return (c_combined, step), c_proj
        return (c_combined, step), c_combined

    def initialize_carry(self, init_key, input_shape):
        # Use the learnable initial carry parameter instead of reinitializing it
        # Return `initial_carry` as `c` and initialize step to 0
        # c = jnp.broadcast_to(self.initial_carry, (input_shape[0], self.context_dim))
        c = jax.random.normal(init_key, (input_shape[0], self.latents)) * 0.3
        # c = jnp.ones((input_shape[0], self.latents)) * self.initial_carry[None, :, 0]
        step = 0
        print("CARRY", c.shape, step)
        return (c, step)
