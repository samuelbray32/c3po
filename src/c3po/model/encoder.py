from typing import Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn

from .util import MLP


class BaseEncoder(nn.Module):
    latent_dim: int

    def setup(self):
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError

    @property
    def requires_random_key(self):
        """Indicates whether the encoder requires a random key for its operations."""
        return False


def encoder_factory(encoder_model: str, latent_dim: int, **kwargs) -> BaseEncoder:
    if encoder_model == "simple":
        return SimpleEncoder(latent_dim=latent_dim, **kwargs)
    if encoder_model == "convolutional1D":
        return convolutionalEncoder1D(latent_dim=latent_dim, **kwargs)
    if encoder_model == "multi_shank_v0":
        return MultiShankEncoderV0(latent_dim=latent_dim, **kwargs)
    if encoder_model == "multi_shank_v1":
        return MultiShankEncoderV1(latent_dim=latent_dim, **kwargs)
    if encoder_model == "sorted_spikes":
        return SortedSpikesEncoder(latent_dim=latent_dim, **kwargs)
    if encoder_model == "identity":
        return IdentityEncoder(latent_dim=latent_dim, **kwargs)
    else:
        raise ValueError(f"Unknown encoder model: {encoder_model}")


class SimpleEncoder(BaseEncoder):
    widths: Sequence[int]

    def setup(self):
        self.encoder = MLP(
            self.widths + (self.latent_dim,), kernel_init=nn.initializers.he_uniform()
        )

    def __call__(self, x):
        return self.encoder(x)


class convolutionalEncoder1D(BaseEncoder):
    conv_kernel_sizes: Sequence[int]
    conv_strides: Sequence[int]
    conv_features: Sequence[int]
    widths: Sequence[int]

    def setup(self):
        if (
            not len(self.conv_kernel_sizes)
            == len(self.conv_strides)
            == len(self.conv_features)
        ):
            raise ValueError(
                "conv_kernel_sizes and conv_strides must have the same length"
            )
        conv_layers = []
        print("HI")
        for i, (kernel_size, stride, features) in enumerate(
            zip(self.conv_kernel_sizes, self.conv_strides, self.conv_features)
        ):
            # if i == 0:
            #     kernel = kernel_size
            # else:
            #     kernel = 1
            print(i)
            conv_layers.append(nn.Conv(features, kernel_size, stride, padding="VALID"))
            # self.conv_layers.append(nn.relu)
        self.conv_layers = conv_layers
        print("HI")
        # self.flatten = nn.Flatten()
        self.dense = MLP(
            self.widths + (self.latent_dim,), kernel_init=nn.initializers.he_uniform()
        )

    def __call__(self, x):
        x = jnp.expand_dims(x, axis=-1)
        for layer in self.conv_layers:
            print(x.shape)
            x = layer(x)

        print(x.shape)
        x = jax.lax.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
        return self.dense(x)


class MultiShankEncoderV0(BaseEncoder):
    """makes a separate encoder model for each shank then combines"""

    n_shanks: int
    shank_encoder_params: dict
    latent_dim: int

    def setup(self):
        self.shank_encoders = [
            encoder_factory(**self.shank_encoder_params, latent_dim=self.latent_dim)
            for _ in range(self.n_shanks)
        ]

    def __call__(self, x):
        print("x shape", x.shape)
        print("x shape", x[..., 0].shape)
        encoded_shanks = [
            encoder(x[..., i]) for i, encoder in enumerate(self.shank_encoders)
        ]
        print("encoded shanks shape", encoded_shanks[0].shape)
        encoded_shanks = jnp.stack(encoded_shanks, axis=-1)
        print("encoded shanks shape", encoded_shanks.shape)
        shank_indicator = jnp.sum(jnp.abs(x), axis=-2)
        print("shank indicator shape", shank_indicator.shape)
        shank_indicator = jnp.where(shank_indicator != 0, 1.0, 0.0)
        print("shank indicator shape", shank_indicator.shape)
        encoded_shanks = jnp.sum(encoded_shanks * shank_indicator[:, None, :], axis=-1)
        # encoded_shanks = jnp.sum(encoded_shanks, axis=-1)
        print("encoded shanks shape", encoded_shanks.shape)
        return encoded_shanks


class MultiShankEncoderV1(BaseEncoder):
    """makes a separate encoder model for each shank then combines
    assumes that the shank indicator is the last entry in the input vector
    """

    n_shanks: int
    shank_encoder_params: dict
    latent_dim: int

    def setup(self):
        self.shank_encoders = [
            encoder_factory(**self.shank_encoder_params, latent_dim=self.latent_dim)
            for _ in range(self.n_shanks)
        ]

        self.expanded_shank_encoders = [
            lambda dummmy_self, x: encoder(x) for encoder in self.shank_encoders
        ]

    def __call__(self, x):
        # x has shape (batch_size, n_channels+1)
        print("x shape", x.shape)
        # Separate the data part (all but last channel) from the integer shank indicator
        x_data = x[..., :-1]  # shape: (batch_size, n_channels)
        x_shank = x[..., -1].astype(int)  # shape: (batch_size,)

        batch_size, _ = x_data.shape

        # outputs = []
        # for x_i, shank_i in zip(x_data, x_shank):
        #     encoded_data = nn.switch(shank_i, self.expanded_shank_encoders, self, x_i)

        #     outputs.append(encoded_data)
        # outputs = jnp.stack(outputs, axis=0)
        # return outputs

        # Prepare an output array to store latent vectors of shape (batch_size, latent_dim)
        outputs = jnp.zeros(
            (batch_size + 1, self.latent_dim), dtype=x_data.dtype
        )  # last index is a garbage slot

        # For each shank index, gather the data belonging to that shank, encode, and scatter back
        for shank_idx in range(self.n_shanks):
            # Gather integer indices where x_shank == shank_idx
            # 'size=batch_size' ensures a fixed-size output for JIT
            idxs = jnp.where(x_shank == shank_idx, size=batch_size, fill_value=-1)[0]

            # Select the corresponding data rows
            data_for_shank = x_data[idxs]  # shape: (num_selected, n_channels)

            # Encode using the shank-specific encoder
            encoded_data = self.shank_encoders[shank_idx](
                data_for_shank
            )  # (num_selected, latent_dim)

            # Scatter the encoded data back
            outputs = outputs.at[idxs].set(encoded_data)
        return outputs[:-1]


class SortedSpikesEncoder(BaseEncoder):
    """Embeds sorted spike data
    input data should be one-hot encoded
    """

    n_units: int
    gauss_noise_std: float = 0.0

    def setup(self):
        self.m = self.param(
            "encoder_matrix",
            # nn.initializers.he_uniform(),
            nn.initializers.normal(stddev=0.1),
            (self.n_units, self.latent_dim),
        )

    def __call__(self, x, rand_key=None):
        if not self.requires_random_key:
            return jnp.dot(x, self.m)
        return jax.random.normal(rand_key, (x.shape[0], self.latent_dim)) + jnp.dot(
            x, self.m
        )

    @property
    def requires_random_key(self):
        """Indicates whether the encoder requires a random key for its operations."""
        return self.gauss_noise_std > 0


class IdentityEncoder(BaseEncoder):

    def setup(self):
        return

    def __call__(self, x):
        if x.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Input shape {x.shape} does not match latent_dim {self.latent_dim}"
            )
        # Simply return the input as is
        return jnp.array(x)
