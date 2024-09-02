from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import scan
from functools import partial


def rate_prediction_factory(
    rate_model: str, latent_dim: int, context_dim: int, **kwargs
):

    if rate_model == "bilinear":
        return BilinearRatePrediction(
            context_dim=context_dim, latent_dim=latent_dim, **kwargs
        )
    else:
        raise ValueError(f"Unknown rate model: {rate_model}")


class BilinearRatePrediction(nn.Module):
    context_dim: int
    latent_dim: int

    @nn.compact
    def __call__(self, z, c):
        # Initialize a trainable matrix W with shape (latent_dim, context_dim)
        W = self.param(
            "W", nn.initializers.lecun_normal(), (self.latent_dim, self.context_dim)
        )
        b = self.param("b", nn.initializers.zeros, (1,))

        # Perform the operation z^T W c
        func = lambda zi, ci: jnp.squeeze(jnp.dot(zi, jnp.dot(W, ci)) + b)
        return jnp.squeeze(jnp.clip(jnp.exp(jax.vmap(func)(z, c)), min=1e-8, max=1e3))
