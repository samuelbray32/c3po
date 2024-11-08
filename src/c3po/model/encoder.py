from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import scan
from functools import partial

from .util import MLP


class BaseEncoder(nn.Module):
    latent_dim: int

    def setup(self):
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError


def encoder_factory(encoder_model: str, latent_dim: int, **kwargs) -> BaseEncoder:
    if encoder_model == "simple":
        return SimpleEncoder(latent_dim=latent_dim, **kwargs)
    if encoder_model == "convolutional1D":
        return convolutionalEncoder1D(latent_dim=latent_dim, **kwargs)
    if encoder_model == "multi_shank":
        return MultiShankEncoder(latent_dim=latent_dim, **kwargs)

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


class MultiShankEncoder(BaseEncoder):
    """makes a seperate encoder model for each shank then combines"""

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
