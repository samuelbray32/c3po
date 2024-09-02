from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import scan
from functools import partial

from .model import MLP


def encoder_factory(encoder_model: str, latent_dim: int, **kwargs) -> BaseEncoder:
    if encoder_model == "simple":
        return SimpleEncoder(widths=widths, latent_dim=latent_dim, **kwargs)
    else:
        raise ValueError(f"Unknown encoder model: {encoder_model}")


class BaseEncoder(nn.Module):
    latent_dim: int

    def setup(self):
        raise NotImplementedError

    def __call__(self, x):
        raise NotImplementedError


class SimpleEncoder(BaseEncoder):
    widths: Sequence[int]

    def setup(self):
        self.encoder = MLP(
            self.widths + (self.latent_dim,), kernel_init=nn.initializers.he_uniform()
        )

    def __call__(self, x):
        return self.encoder(x)
