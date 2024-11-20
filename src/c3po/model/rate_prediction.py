from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import scan
from functools import partial
from .util import MLP


def rate_prediction_factory(
    rate_model: str, latent_dim: int, context_dim: int, **kwargs
):

    if rate_model == "bilinear":
        return BilinearRatePrediction(
            context_dim=context_dim, latent_dim=latent_dim, **kwargs
        )
    elif rate_model == "denseBilinear":
        return DenseBilinearRatePrediction(
            context_dim=context_dim, latent_dim=latent_dim, **kwargs
        )
    elif rate_model == "dense":
        return DenseRatePrediction(**kwargs)
    elif rate_model == "bilinearLinear":
        return BilinearLinearRatePrediction(
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

class DenseBilinearRatePrediction(nn.Module):
    context_dim: int
    latent_dim: int
    widths: Sequence[int]

    @nn.compact
    def __call__(self, z, c):
        # Initialize a trainable matrix W with shape (latent_dim, context_dim)
        c = MLP(
            self.widths, kernel_init=nn.initializers.he_uniform()
        )(c)
        W = self.param(
            "W", nn.initializers.lecun_normal(), (self.latent_dim, self.widths[-1])
        )
        b = self.param("b", nn.initializers.zeros, (1,))
        # Perform the operation z^T W c
        func = lambda zi, ci: jnp.squeeze(jnp.dot(zi, jnp.dot(W, ci)) + b)
        return jnp.squeeze(jnp.clip(jnp.exp(jax.vmap(func)(z, c)), min=1e-8, max=1e3))

class BilinearLinearRatePrediction(nn.Module):
    context_dim: int
    latent_dim: int

    @nn.compact
    def __call__(self, z, c):
        # Initialize a trainable matrix W with shape (latent_dim, context_dim)
        W = self.param(
            "W", nn.initializers.lecun_normal(), (self.latent_dim, self.context_dim)
        )
        b = self.param("b", nn.initializers.zeros, (1,))
        W_c = self.param(
            "W_c", nn.initializers.lecun_normal(), (self.context_dim, 1)
        )

        # Perform the operation z^T W c
        func = lambda zi, ci: jnp.squeeze(jnp.dot(zi, jnp.dot(W, ci)) + b + jnp.dot(ci, W_c))
        return jnp.squeeze(jnp.clip(jnp.exp(jax.vmap(func)(z, c)), min=1e-8, max=1e3))


class DenseRatePrediction(nn.Module):
    widths: Sequence[int]

    @nn.compact
    def __call__(self, z, c):
        q = jnp.concatenate([z, c], axis=-1)
        print(z.shape,q.shape)
        log_r = MLP(self.widths + (1,), kernel_init=nn.initializers.he_uniform())(q)
        return jnp.squeeze(jnp.clip(jnp.exp(log_r), min=1e-8, max=1e3))