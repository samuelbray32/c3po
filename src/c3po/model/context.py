from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial

from .util import (
    MLP,
    StabilizedDenseDynamics,
    DilatedCausalConv1D,
    CausalConv1D,
    causal_smoothing,
)


def context_factory(context_model: str, context_dim: int, **kwargs):
    kwargs["infer_init"] = kwargs.get("infer_init", False)
    if context_model == "simpleRNN":
        return SimpleRNNContext(context_dim=context_dim, **kwargs)
    elif context_model == "stableSimpleRNN":
        return StableSimpleRNNContext(context_dim=context_dim, **kwargs)
    elif context_model == "fastSlowRNN":
        return FastSlowRNNContext(context_dim=context_dim, **kwargs)
    elif context_model == "heirarchicalRNN":
        return HeirarchicalRNN(context_dim=context_dim, **kwargs)
    elif context_model == "lfads":
        return LFADSContext(context_dim=context_dim, **kwargs)
    elif context_model == "LSTM":
        return nn.LSTMCell(name="context", features=context_dim, **kwargs)
    elif context_model == "projectedLSTM":
        return ProjectedLSTM(context_dim=context_dim, **kwargs)
    elif context_model == "GRU":
        return nn.GRUCell(name="context", features=context_dim, **kwargs)
    elif context_model == "myGRU":
        return MyGRU(context_dim=context_dim, **kwargs)
    elif context_model == "nullRNN":
        return NullRNN(context_dim=context_dim, **kwargs)
    elif context_model == "wavenet":
        kwargs.pop("infer_init", None)
        return Wavenet(context_dim=context_dim, **kwargs)
    elif context_model == "wavenet_v2":
        kwargs.pop("infer_init", None)
        return nn.remat(WavenetV2)(context_dim=context_dim, **kwargs)
    elif context_model == "causalTransformer":
        kwargs.pop("infer_init", None)  # not an RNN cell
        return CausalTransformer(context_dim=context_dim, **kwargs)
    elif context_model == "samsTransformer":
        kwargs.pop("infer_init", None)
        return SamsCausalTransformer(context_dim=context_dim, **kwargs)
    elif context_model == "slidingWindow":
        kwargs.pop("infer_init", None)
        return SlidingWindow(context_dim=context_dim, **kwargs)
    else:
        raise ValueError(f"Unknown context model: {context_model}")


class BaseContext(nn.RNNCellBase):
    context_dim: int
    infer_init: bool

    def setup(self):
        if self.infer_init:
            self.infer_carry = DefaultInferCarry(self.context_dim)

    def __call__(self, c, z):
        raise NotImplementedError

    def initialize_carry(self, init_key, input_shape):
        raise NotImplementedError

    def initialize_carry_from_data(self, z):
        # raise NotImplementedError
        return self.infer_carry(z)

    @property
    def num_feature_axes(self):
        return 1


class DefaultInferCarry(nn.Module):
    context_dim: int

    def setup(self):
        self._forward = nn.GRUCell(features=30)
        self._backward = nn.GRUCell(features=30)
        self.forward = nn.RNN(self._forward, time_major=False)
        self.backward = nn.RNN(self._backward, time_major=False, reverse=True)
        self.dense = MLP([32, 32, self.context_dim])

    def __call__(self, z):
        z_forward = self.forward(z)[:, -1, :]
        z_backward = self.backward(z)[:, -1, :]
        features = jnp.concatenate([z_forward, z_backward], axis=-1)
        return self.dense(features)


class NullRNN(BaseContext):
    def __call__(self, c, z):
        return 0, z

    def initialize_carry(self, init_key, input_shape):
        return 0


class MyGRU(BaseContext):
    gru_dim: int

    def setup(self):
        self.gru = nn.GRUCell(name="gru_context", features=self.gru_dim)
        self.w = self.param(
            "w", nn.initializers.xavier_uniform(), (self.gru_dim, self.context_dim)
        )

    def __call__(self, carry, z):
        new_carry, new_state = self.gru(carry, z)
        return new_carry, jnp.dot(new_state, self.w)

    def initialize_carry(self, init_key, input_shape):
        return self.gru.initialize_carry(init_key, input_shape)


class layeredGRU(BaseContext):
    gru_dims: Sequence[int]

    def setup(self):
        self.gru = [
            nn.GRUCell(name=f"gru_layer_{i}", features=dim)
            for i, dim in enumerate(self.gru_dims)
        ]
        self.w = self.param(
            "w", nn.initializers.xavier_uniform(), (self.gru_dims[-1], self.context_dim)
        )

    def __call__(self, carry, z):
        state_list = [z]
        carry_list = []
        for i, gru in enumerate(self.gru):
            new_state, new_carry = gru(carry[i], state_list[i])
            state_list.append(new_state)
            carry_list.append(new_carry)
        return carry_list, jnp.dot(state_list[-1], self.w)

    def initialize_carry(self, init_key, input_shape):
        layer_inputs = [input_shape]
        for gru in self.gru[:-1]:
            layer_inputs.append((input_shape[0], gru.features))


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
    carry_stabilized: bool = False

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
        if self.carry_stabilized:
            c_combined, step, stable_matrices = carry
        else:
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
            if self.carry_stabilized:
                c_fast_trans, stable_fast = self.history_matrix_fast(
                    c_fast, stable_matrices[0]
                )
            else:
                c_fast_trans = self.history_matrix_fast(c_fast)
        else:
            c_fast_trans = c_fast - self.history_matrix_fast(c_fast)
        c_fast_new = z_trans + c_fast_trans

        # Slow state update only at specified intervals
        update_slow = nn.relu(step - self.slow_update_freq + 1)
        c_fast_slow = self.fast_to_slow(c_fast)
        if self.stabilize_dynamics:
            if self.carry_stabilized:
                c_slow_trans, stable_slow = self.history_matrix_slow(
                    c_slow, stable_matrices[1]
                )
            else:
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
        # make the new carry
        if self.carry_stabilized:
            new_carry = (c_combined, step, (stable_fast, stable_slow))
        else:
            new_carry = (c_combined, step)

        if self.latents != self.context_dim:
            c_proj = self.context_projection(c_combined)
            return new_carry, c_proj
        return new_carry, c_combined

    def initialize_carry(self, init_key, input_shape):
        # Use the learnable initial carry parameter instead of reinitializing it
        # Return `initial_carry` as `c` and initialize step to 0
        # c = jnp.broadcast_to(self.initial_carry, (input_shape[0], self.context_dim))
        # c = jax.random.normal(init_key, (input_shape[0], self.latents)) * 0.3
        c = (
            jnp.ones((input_shape[0], self.latents)) * self.initial_carry.T
        )  # [None, :, 0]
        step = 0
        # print("CARRY", c.shape, step)
        # if self.carry_stabilized:
        #     stable_fast = self.history_matrix_fast.stable_matrix()
        #     stable_slow = self.history_matrix_slow.stable_matrix()
        #     return (c, step, (stable_fast, stable_slow))
        return (c, step)


class ProjectedLSTM(BaseContext):
    expanded_dim: Sequence[int]
    update_freq: Sequence[int]

    def setup(self):
        self.lstm = [
            nn.LSTMCell(name=f"lstm_layer_{i}", features=dim)
            for i, dim in enumerate(self.expanded_dim)
        ]
        self.project = nn.Dense(
            self.context_dim,
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(),
        )

    def __call__(self, carry, z):
        carry_list, step_list = carry
        inputs_list = [z] + [c[1] for c in carry_list[:-1]]

        new_carry_list = []
        new_state_list = []
        new_step_list = []

        for carry_i, layer_input, lstm, freq, step in zip(
            carry_list, inputs_list, self.lstm, self.update_freq, step_list
        ):
            update = nn.relu(step - freq + 1)
            step = (step) * (1 - update) + 1
            new_carry, new_state = lstm(carry_i, layer_input)
            new_carry = new_carry[0]
            # new_state = jnp.clip(new_state, -1000, 1000)

            # only update the state and carry if the update flag is on
            new_state = new_state * update + carry_i[1] * (1 - update)
            new_carry = (new_carry * update + carry_i[0] * (1 - update), new_state)
            # log it
            new_step_list.append(step)
            new_carry_list.append(new_carry)
            new_state_list.append(new_state)

        new_state = jnp.concatenate(new_state_list, axis=-1)

        print("NEWSTATE", new_state.shape)
        c = self.project(new_state)
        return (new_carry_list, step_list), c

    def initialize_carry(self, init_key, input_shape):
        # return self.lstm.initialize_carry(init_key, input_shape)
        return (
            [lstm.initialize_carry(init_key, input_shape) for lstm in self.lstm],
            [0 for _ in self.lstm],
        )


class HeirarchicalRNN(BaseContext):
    scales: Sequence[float]
    expanded_dim: Sequence[int]
    update_freq: Sequence[int]

    def setup(self):
        super().setup()

        # dynamics of each RNN layer
        self.dynamics = [
            StabilizedDenseDynamics(
                dim,
                delta_matrix=False,
                kernel_init=partial(
                    timescale_initializer,
                    scales=list(np.abs(np.random.normal(s, 0.3, dim))),
                ),
            )
            for dim, s in zip(self.expanded_dim, self.scales)
        ]
        # connection up the RNN layers
        self.pass_through = [
            nn.Dense(dim, use_bias=False) for dim in self.expanded_dim[1:]
        ]
        # from z to RNN_1
        self.input_network = nn.Dense(self.expanded_dim[0])

        # project the final state to the context
        self.project = nn.Dense(
            self.context_dim, use_bias=False, kernel_init=nn.initializers.orthogonal()
        )

        if self.infer_init:
            self.dense_init = [MLP([16, 16, d]) for d in self.expanded_dim]
        else:
            self.initial_carry = [
                self.param(
                    f"initial_carr_{i}",  # Parameter name
                    lambda key: jax.random.normal(key, (1, dim)) * 3,
                )
                for i, dim in enumerate(self.expanded_dim)
            ]

    def __call__(self, carry, z):
        state_list, step_list = carry
        print("STATE", [s.shape for s in state_list])
        print("STEP", step_list)

        # update the first layer
        new_step_list = [step_list[0]]
        new_state_list = [
            # jnp.tanh(self.input_network(z))
            self.input_network(z)
            + self.dynamics[0](state_list[0])
        ]
        # update the rest of the layers
        for i in range(1, len(self.dynamics)):
            new_state = self.dynamics[i](state_list[i]) + self.pass_through[i - 1](
                state_list[i - 1]
            )

            update = nn.relu(step_list[i] - self.update_freq[i] + 1)
            new_step_list.append((step_list[i]) * (1 - update) + 1)
            new_state = new_state * update + state_list[i] * (1 - update)
            new_state_list.append(new_state)

        combined_state = jnp.concatenate(new_state_list, axis=-1)
        combined_state = jnp.clip(combined_state, -1000, 1000)
        c = self.project(combined_state)
        return (new_state_list, new_step_list), c

    def initialize_carry(self, init_key, input_shape):
        # state = [
        #     # jnp.broadcast_to(
        #     #     self.initial_carry[i], (input_shape[0], self.expanded_dim[i])
        #     # )
        #     jnp.ones((input_shape[0], self.expanded_dim[i])) * self.initial_carry[i][0]
        #     for i in range(len(self.expanded_dim))
        # ]
        state = [
            jnp.broadcast_to(
                self.initial_carry[i], (input_shape[0], self.expanded_dim[i])
            )
            for i in range(len(self.expanded_dim))
        ]

        print("STATE", [s.shape for s in state])
        alt = [
            jax.random.normal(init_key, (input_shape[0], dim)) * 0.3
            for dim in self.expanded_dim
        ]
        print("ALT", [s.shape for s in alt])
        return (state, [0 for _ in self.dynamics])
        # return (
        #     [
        #         jax.random.normal(init_key, (input_shape[0], dim)) * 0.3
        #         for dim in self.expanded_dim
        #     ],
        #     [0 for _ in self.dynamics],
        # )

    def initialize_carry_from_data(self, z):
        inferred_val = super().initialize_carry_from_data(z)
        state = [dense(inferred_val) for dense in self.dense_init]
        return state, [0 for _ in self.dynamics]


class SlidingWindow(nn.Module):
    """Simple model that applies a causal smoothing filter over the input marks
    Requires that context and latent space arer same dimension. Used for rapid training
    architecture
    """

    context_dim: int
    window_size: int
    smoothing_decay: float = (
        None  # if not None, causal smoothing filter has exponential decay
    )

    def __call__(self, x):
        return causal_smoothing(
            x[..., :-1], self.window_size, decay=self.smoothing_decay
        )


class Wavenet(nn.Module):
    """Wavenet Model.
    Citation: https://arxiv.org/pdf/1609.03499
    """

    layer_dilations: Sequence[int]
    layer_kernel_size: Sequence[int]
    context_dim: int  # Number of output channels
    expanded_dim: int
    smoothing: int = (
        1  # if >1, applies a square filter over the input marks over time to smooth fluctuation in input
    )
    smoothing_decay: float = (
        None  # if not None, causal smoothing filter has exponential decay
    )
    categorical: bool = False  # if true , output is softmaxed
    residual_model: bool = (
        False  # if true, cumsum the final output. Wavenet is then learning residuals at each timestep
    )
    final_smooth: int = 0  # causal smoothing filter on final output

    def setup(self):
        self.tanh_layers = [
            DilatedCausalConv1D(
                features=self.expanded_dim,
                kernel_size=self.layer_kernel_size[i],
                dilation=self.layer_dilations[i],
                use_bias=True,
            )
            for i in range(len(self.layer_dilations))
        ]
        self.gating_layers = [
            DilatedCausalConv1D(
                features=self.expanded_dim,
                kernel_size=self.layer_kernel_size[i],
                dilation=self.layer_dilations[i],
                use_bias=True,
            )
            for i in range(len(self.layer_dilations))
        ]
        self.residual_layers = [
            CausalConv1D(features=self.context_dim, kernel_size=1, use_bias=False)
            for _ in range(len(self.layer_dilations))
        ]
        self.skip_connections = [
            CausalConv1D(features=self.context_dim, kernel_size=1, use_bias=True)
            for _ in range(len(self.layer_dilations))
        ]

        self.initial_layer = CausalConv1D(
            self.context_dim, kernel_size=1, use_bias=False
        )

        self.post_sum = CausalConv1D(self.context_dim * 4, kernel_size=1, use_bias=True)
        self.final_proj = CausalConv1D(self.context_dim, kernel_size=1, use_bias=False)

    def __call__(self, x):
        if self.smoothing > 1:
            x = causal_smoothing(x, self.smoothing, decay=self.smoothing_decay)
        x = self.initial_layer(x)  # expand from latent to context dimension
        skip_connection = jnp.zeros(x.shape)
        for tanh_layer, gating_layer, residual_layer, skip_layer in zip(
            self.tanh_layers,
            self.gating_layers,
            self.residual_layers,
            self.skip_connections,
        ):
            delta = jnp.tanh(tanh_layer(x)) * jax.nn.sigmoid(
                gating_layer(x)
            )  # Residual Gated activation units
            skip_connection = skip_connection + skip_layer(delta)
            delta = residual_layer(delta)
            x = x + delta

        if self.residual_model:
            print("RESIDUAL", x.shape)
            x = jnp.cumsum(skip_connection, axis=-2)  # cumsum over time

        else:
            skip_connection = nn.relu(skip_connection)
            x = self.post_sum(skip_connection)
            x = jax.nn.relu(x)
            x = self.final_proj(x)
        if self.categorical:
            x = jax.nn.softmax(x, axis=-1)

        if self.final_smooth > 1:
            x = causal_smoothing(x, self.final_smooth)

        return x


class WavenetV2(nn.Module):
    """Wavenet Model.
    Citation: https://arxiv.org/pdf/1609.03499

    TODO: V2 Unused, deprecated in future release
    """

    layer_dilations: Sequence[int]
    layer_kernel_size: Sequence[int]
    context_dim: int  # Number of output channels
    latent_dim: int  # Number of input channels
    expanded_dim: int
    smoothing: int = (
        1  # if >1, applies a square filter over the input marks over time to smooth fluctuation in input
    )
    categorical: bool = False  # if true , output is softmaxed
    residual_model: bool = (
        False  # if true, cumsum the final output. Wavenet is then learning residuals at each timestep
    )
    final_smooth: int = 0  # causal smoothing filter on final output

    def setup(self):
        self.tanh_layers = [
            DilatedCausalConv1D(
                features=self.expanded_dim,
                kernel_size=self.layer_kernel_size[i],
                dilation=self.layer_dilations[i],
                use_bias=True,
            )
            for i in range(len(self.layer_dilations))
        ]
        self.gating_layers = [
            DilatedCausalConv1D(
                features=self.expanded_dim,
                kernel_size=self.layer_kernel_size[i],
                dilation=self.layer_dilations[i],
                use_bias=True,
            )
            for i in range(len(self.layer_dilations))
        ]
        self.residual_layers = [
            CausalConv1D(features=self.latent_dim, kernel_size=1, use_bias=False)
            for _ in range(len(self.layer_dilations))
        ]
        self.skip_connections = [
            CausalConv1D(features=self.latent_dim, kernel_size=1, use_bias=True)
            for _ in range(len(self.layer_dilations))
        ]

        # self.initial_layer = CausalConv1D(
        #     self.context_dim, kernel_size=1, use_bias=False
        # )

        self.post_sum = CausalConv1D(self.context_dim * 4, kernel_size=1, use_bias=True)
        self.final_proj = CausalConv1D(self.context_dim, kernel_size=1, use_bias=False)

    def __call__(self, x):
        if self.smoothing > 1:
            x = causal_smoothing(x, self.smoothing)
        # x = self.initial_layer(x)  # expand from latent to context dimension
        skip_connection = jnp.zeros(x.shape)
        for tanh_layer, gating_layer, residual_layer, skip_layer in zip(
            self.tanh_layers,
            self.gating_layers,
            self.residual_layers,
            self.skip_connections,
        ):
            delta = jnp.tanh(tanh_layer(x)) * jax.nn.sigmoid(
                gating_layer(x)
            )  # Residual Gated activation units
            skip_connection = skip_connection + skip_layer(delta)
            delta = residual_layer(delta)
            x = x + delta

        if self.residual_model:
            print("RESIDUAL", x.shape)
            x = jnp.cumsum(skip_connection, axis=-2)  # cumsum over time

        else:
            skip_connection = nn.relu(skip_connection)
            x = self.post_sum(skip_connection)
            x = jax.nn.relu(x)
            x = self.final_proj(x)
        if self.categorical:
            x = jax.nn.softmax(x, axis=-1)

        if self.final_smooth > 1:
            x = causal_smoothing(x, self.final_smooth)

        return x

    # class WavenetV2(nn.Module):
    #     """Wavenet Model.
    #     Citation: https://arxiv.org/pdf/1609.03499

    #     Motivations:
    #     - Now assume dim(Z) > dim(C)

    #     Changes from V1:
    #     - concatenate the skip connections instead of sum
    #     - maintain latent dim rather than context dim through dilated conv layers
    #     - remove the initial layer (since we are keeping the latent dim)

    #     """

    #     layer_dilations: Sequence[int]
    #     layer_kernel_size: Sequence[int]
    #     latent_dim: int  # Number of input channels
    #     context_dim: int  # Number of output channels
    #     expanded_dim: int
    #     smoothing: int = (
    #         1  # if >1, applies a square filter over the input marks over time to smooth fluctuation in input
    #     )
    #     categorical: bool = False  # if true , output is softmaxed
    #     residual_model: bool = (
    #         False  # if true, cumsum the final output. Wavenet is then learning residuals at each timestep
    #     )
    #     final_smooth: int = 0  # causal smoothing filter on final output

    #     def setup(self):
    #         self.tanh_layers = [
    #             DilatedCausalConv1D(
    #                 features=self.expanded_dim,
    #                 kernel_size=self.layer_kernel_size[i],
    #                 dilation=self.layer_dilations[i],
    #                 use_bias=True,
    #             )
    #             for i in range(len(self.layer_dilations))
    #         ]
    #         self.gating_layers = [
    #             DilatedCausalConv1D(
    #                 features=self.expanded_dim,
    #                 kernel_size=self.layer_kernel_size[i],
    #                 dilation=self.layer_dilations[i],
    #                 use_bias=True,
    #             )
    #             for i in range(len(self.layer_dilations))
    #         ]
    #         self.residual_layers = [
    #             CausalConv1D(features=self.latent_dim, kernel_size=1, use_bias=False)
    #             for _ in range(len(self.layer_dilations))
    #         ]
    #         self.skip_connections = [
    #             CausalConv1D(features=self.context_dim, kernel_size=1, use_bias=True)
    #             for _ in range(len(self.layer_dilations))
    #         ]

    #         # self.initial_layer = CausalConv1D(
    #         #     self.context_dim, kernel_size=1, use_bias=False
    #         # )

    #         self.post_sum = CausalConv1D(self.context_dim * 4, kernel_size=1, use_bias=True)
    #         self.final_proj = CausalConv1D(self.context_dim, kernel_size=1, use_bias=False)

    #     def __call__(self, x):
    #         if self.smoothing > 1:
    #             x = causal_smoothing(x, self.smoothing)
    #         # x = self.initial_layer(x)  # expand from latent to context dimension
    #         skip_connection = []
    #         for tanh_layer, gating_layer, residual_layer, skip_layer in zip(
    #             self.tanh_layers,
    #             self.gating_layers,
    #             self.residual_layers,
    #             self.skip_connections,
    #         ):
    #             delta = jnp.tanh(tanh_layer(x)) * jax.nn.sigmoid(
    #                 gating_layer(x)
    #             )  # Residual Gated activation units
    #             skip_connection.append(skip_layer(delta))
    #             delta = residual_layer(delta)
    #             x = x + delta

    #         if self.residual_model:
    #             raise ValueError(
    #                 "Poor performance on testing of residual model. Slated for removal"
    #             )
    #             print("RESIDUAL", x.shape)
    #             x = jnp.cumsum(skip_connection, axis=-2)  # cumsum over time

    #         else:
    #             skip_connection = jnp.concatenate(skip_connection, axis=-1)
    #             x = self.post_sum(skip_connection)
    #             x = jax.nn.relu(x)
    #             x = self.final_proj(x)
    #         if self.categorical:
    #             x = jax.nn.softmax(x, axis=-1)

    #         if self.final_smooth > 1:
    #             x = causal_smoothing(x, self.final_smooth)

    #         return x


"""
CAUSAL TRANSFORMER
"""
# causal_transformer.py
from typing import Sequence, Tuple, Any
import jax.numpy as jnp
import flax.linen as nn
from .util import causal_smoothing  # already in your repo


def _split_heads(x: jnp.ndarray, num_heads: int) -> jnp.ndarray:
    B, T, D = x.shape
    d_k = D // num_heads
    return x.reshape(B, T, num_heads, d_k).transpose(0, 2, 1, 3)  # [B,H,T,d_k]


def _merge_heads(x: jnp.ndarray) -> jnp.ndarray:
    B, H, T, d_k = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, H * d_k)  # [B,T,D]


class FlashSelfAttention(nn.Module):
    d_model: int
    num_heads: int
    dropout: float = 0.0
    dtype: Any = jnp.float32  # parameter & output dtype
    attn_dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x, *, deterministic: bool):
        # 1) project to QKV
        qkv = nn.Dense(3 * self.d_model, use_bias=False, name="qkv")(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # # 2) split heads
        # q = _split_heads(q, self.num_heads).astype(self.attn_dtype)
        # k = _split_heads(k, self.num_heads).astype(self.attn_dtype)
        # v = _split_heads(v, self.num_heads).astype(self.attn_dtype)

        q = q.astype(self.attn_dtype)
        k = k.astype(self.attn_dtype)
        v = v.astype(self.attn_dtype)

        # 3) flash attention kernel (cuDNN / Triton / Pallas)
        attn_out = jax.nn.dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            # dropout=self.dropout if not deterministic else 0.0,
            # deterministic=deterministic,
            implementation="cudnn",  # ⇐ pick "pallas" / "flash_attention" if needed
        )  # [B,H,T,d_k]
        attn_out = attn_out.astype(self.dtype)

        # # 4) merge heads & final projection
        # attn_out = _merge_heads(attn_out).astype(self.dtype)
        out = nn.Dense(self.d_model, use_bias=False, name="out_proj")(attn_out)
        return out


class FlashTransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    mlp_dim: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, mask, deterministic: bool):
        # --- Flash Attention path -------------------------------------------------
        h = nn.LayerNorm(name="ln1")(x)
        h = FlashSelfAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
            name="flash_attn",
        )(h, deterministic=deterministic)
        x = x + h  # residual‑1

        # --- Feed‑forward ---------------------------------------------------------
        h = nn.LayerNorm(name="ln2")(x)
        h = nn.Dense(self.mlp_dim, name="fc1")(h)
        h = nn.gelu(h)
        h = nn.Dropout(rate=self.dropout)(h, deterministic=deterministic)
        h = nn.Dense(self.d_model, name="fc2")(h)
        x = x + h  # residual‑2
        return x


class _TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    mlp_dim: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, mask, *, deterministic: bool):
        h = nn.LayerNorm()(x)
        h = nn.SelfAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout,
            deterministic=deterministic,
        )(h, mask=mask)
        x = x + h  # residual 1

        h = nn.LayerNorm()(x)
        h = nn.Dense(self.mlp_dim)(h)
        h = nn.gelu(h)
        h = nn.Dropout(self.dropout)(h, deterministic=deterministic)
        h = nn.Dense(self.d_model)(h)
        x = x + h  # residual 2
        return x


class CausalTransformer(nn.Module):
    """Causal‑masked Transformer for context inference."""

    num_layers: int
    num_heads: int
    d_model: int
    mlp_dim: int
    context_dim: int
    max_len: int = 2048
    dropout: float = 0.0
    final_smooth: int = 0  # optional causal smoothing of output
    categorical: bool = False  # softmax on channel axis if True

    def setup(self):
        # project latent marks to model dimension
        self.in_proj = nn.Dense(self.d_model, use_bias=False)
        # learnable absolute positional embeddings
        self.pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (self.max_len, self.d_model),
        )
        self.blocks = [
            FlashTransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
            )
            for _ in range(self.num_layers)
        ]
        self.out_proj = nn.Dense(self.context_dim, use_bias=False)

    def __call__(self, z, *, deterministic: bool = True):
        """
        Args
        ----
        z : float32[batch, time, latent_dim]
            Input mark sequence.
        Returns
        -------
        c : float32[batch, time, context_dim]
            Causal context embedding at every time‑step.
        """
        x = self.in_proj(z)
        seq_len = x.shape[1]
        # add / slice positional embeddings
        x = x + self.pos_embed[:seq_len]

        # build causal attention mask once
        mask = nn.make_causal_mask(jnp.ones((seq_len,)), dtype=x.dtype)

        for blk in self.blocks:
            x = blk(x, mask, deterministic=deterministic)

        c = self.out_proj(x)
        if self.categorical:
            c = nn.softmax(c, axis=-1)
        if self.final_smooth > 1:
            c = causal_smoothing(c, self.final_smooth)
        return c


class SamsTransformerBlock(nn.Module):
    qkv_dim: int
    num_heads: int
    mlp_dim: int
    attention_dropout_rate: float = 0.0
    dropout_rate: float = 0.0
    deterministic: bool = True

    @staticmethod
    def flash_attention(q, k, v, **kwargs):
        q = q.astype(jnp.bfloat16)
        k = k.astype(jnp.bfloat16)
        v = v.astype(jnp.bfloat16)
        return jax.nn.dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            implementation="cudnn",  # ⇐ pick "pallas" / "flash_attention" if needed
            # module=module,
        ).astype(jnp.float32)

    @nn.compact
    def __call__(self, inputs, deterministic):
        """Applies Encoder1DBlock module.

        Args:
        inputs: input data.
        deterministic: if true dropout is applied otherwise not.

        Returns:
        output after transformer encoder block.
        """
        # Attention block.
        assert inputs.ndim == 3
        x = nn.LayerNorm()(inputs)

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            # dtype=config.dtype,
            qkv_features=self.qkv_dim,
            # kernel_init=config.kernel_init,
            # bias_init=config.bias_init,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=self.attention_dropout_rate,
            deterministic=deterministic,
            attention_fn=self.flash_attention,
        )(x)

        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + inputs

        # MLP block.
        y = nn.LayerNorm()(x)
        y = nn.Dense(
            self.mlp_dim,
        )(y)
        y = nn.elu(y)
        y = nn.Dense(
            x.shape[-1],
        )(y)
        return x + y


class SamsCausalTransformer(nn.Module):
    """Causal‑masked Transformer for context inference."""

    num_layers: int
    num_heads: int
    d_model: int
    mlp_dim: int
    context_dim: int
    dropout: float = 0.0
    final_smooth: int = 0  # optional causal smoothing of output
    categorical: bool = False  # softmax on channel axis if True
    positional_periods: Sequence[float] = None  # Frequencies for positional encoding
    attention_dropout_rate: float = 0.0

    def setup(self):
        # project latent marks to model dimension
        # self.in_proj = nn.Dense(self.d_model, use_bias=False)
        # learnable absolute positional embeddings

        self.out_proj = nn.Dense(self.context_dim, use_bias=False)

        self.blocks = [
            SamsTransformerBlock(
                qkv_dim=self.d_model * self.num_heads,
                num_heads=self.num_heads,
                attention_dropout_rate=self.attention_dropout_rate,
                dropout_rate=self.dropout,
                mlp_dim=self.mlp_dim,
            )
            for _ in range(self.num_layers)
        ]

        self.default_positional_periods = [
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
        ]

    def _make_positional_embedding(self, delta_t) -> jnp.ndarray:

        t = jnp.cumsum(delta_t, axis=1)
        t = jnp.expand_dims(t, axis=-1)
        sin_embedding = [
            jnp.sin(t / period / (2 * jnp.pi))
            for period in (self.positional_periods or self.default_positional_periods)
        ]
        sin_embedding = jnp.concatenate(sin_embedding, axis=-1)
        cos_embedding = [
            jnp.cos(t / period / (2 * jnp.pi))
            for period in (self.positional_periods or self.default_positional_periods)
        ]
        cos_embedding = jnp.concatenate(cos_embedding, axis=-1)
        embedding = jnp.concatenate([sin_embedding, cos_embedding], axis=-1)
        return embedding

    def __call__(self, z, *, deterministic: bool = True):
        """
        Args
        ----
        z : float32[batch, time, latent_dim]
            Input mark sequence.
        Returns
        -------
        c : float32[batch, time, context_dim]
            Causal context embedding at every time‑step.
        """
        # add / slice positional embeddings
        pos_embedding = self._make_positional_embedding(z[..., -1])
        x = jnp.concatenate([z, pos_embedding], axis=-1)

        # apply transformer layers
        for blk in self.blocks:
            x = blk(x, deterministic=deterministic)

        c = self.out_proj(x)
        if self.categorical:
            c = nn.softmax(c, axis=-1)
        if self.final_smooth > 1:
            c = causal_smoothing(c, self.final_smooth)
        return c
