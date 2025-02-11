from typing import Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn
from .util import MLP
from jax.tree_util import Partial


def rate_prediction_factory(
    rate_model: str, latent_dim: int, context_dim: int, n_params: int, **kwargs
):

    if rate_model == "bilinear":
        return BilinearRatePrediction(
            context_dim=context_dim,
            latent_dim=latent_dim,
            n_params=n_params,
            **kwargs,
        )
    elif rate_model == "denseBilinear":
        return DenseBilinearRatePrediction(
            context_dim=context_dim,
            latent_dim=latent_dim,
            n_params=n_params,
            **kwargs,
        )
    elif rate_model == "dense":
        return DenseRatePrediction(n_params=n_params, **kwargs)
    elif rate_model == "bilinearLinear":
        return BilinearLinearRatePrediction(
            context_dim=context_dim,
            latent_dim=latent_dim,
            n_params=n_params,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown rate model: {rate_model}")


class BilinearRatePrediction(nn.Module):
    context_dim: int  # The dimension of the context vector
    latent_dim: int  # The dimension of the latent vector
    n_params: int  # The number of parameters in the distribution model

    @nn.compact
    def __call__(self, z, c):
        # Initialize a trainable matrix W with shape (latent_dim, context_dim)
        W = [
            self.param(
                f"W_{i}",
                nn.initializers.lecun_normal(),
                (self.latent_dim, self.context_dim),
            )
            for i in range(self.n_params)
        ]
        b = [
            self.param(f"b_{i}", nn.initializers.zeros, (1,))
            for i in range(self.n_params)
        ]

        # Perform the operation z^T W c + b for each parameter
        return jnp.concat(
            [
                jax.vmap(Partial(self.bilinear, W_i, b_i))(z, c)
                for W_i, b_i in zip(W, b)
            ],
            axis=-1,
        )

    @staticmethod
    def bilinear(W, b, z, c):
        return jnp.dot(z, jnp.dot(W, c)) + b


class DenseBilinearRatePrediction(nn.Module):
    context_dim: int
    latent_dim: int
    widths: Sequence[int]

    @nn.compact
    def __call__(self, z, c):
        raise NotImplementedError("Not converted to multiple parameters")
        # # Initialize a trainable matrix W with shape (latent_dim, context_dim)
        # c = MLP(self.widths, kernel_init=nn.initializers.he_uniform())(c)
        # W = self.param(
        #     "W", nn.initializers.lecun_normal(), (self.latent_dim, self.widths[-1])
        # )
        # b = self.param("b", nn.initializers.zeros, (1,))
        # # Perform the operation z^T W c
        # func = lambda zi, ci: jnp.squeeze(jnp.dot(zi, jnp.dot(W, ci)) + b)
        # return jnp.squeeze(jnp.clip(jnp.exp(jax.vmap(func)(z, c)), min=1e-8, max=1e3))


class BilinearLinearRatePrediction(nn.Module):
    context_dim: int
    latent_dim: int
    n_params: int

    @nn.compact
    def __call__(self, z, c):
        # Initialize parameters for each output
        W = [
            self.param(
                f"W_{i}",
                nn.initializers.lecun_normal(),
                (self.latent_dim, self.context_dim),
            )
            for i in range(self.n_params)
        ]
        b = [
            self.param(f"b_{i}", nn.initializers.zeros, (1,))
            for i in range(self.n_params)
        ]
        W_c = [
            self.param(
                f"W_c_{i}", nn.initializers.lecun_normal(), (self.context_dim, 1)
            )
            for i in range(self.n_params)
        ]

        # Perform the operation z^T W c + W_c c + b for each parameter
        return jnp.concat(
            [
                jax.vmap(Partial(self.bilinear_linear, W_i, b_i, W_c_i))(z, c)
                for W_i, b_i, W_c_i in zip(W, b, W_c)
            ],
            axis=-1,
        )

    @staticmethod
    def bilinear_linear(W, b, W_c, z, c):
        return jnp.dot(z, jnp.dot(W, c)) + b + jnp.dot(c, W_c)


class DenseRatePrediction(nn.Module):
    widths: Sequence[int]
    n_params: int

    @nn.compact
    def __call__(self, z, c):
        q = jnp.concatenate([z, c], axis=-1)
        print(z.shape, q.shape)
        log_r = MLP(
            self.widths + (self.n_params,), kernel_init=nn.initializers.he_uniform()
        )(q)
        # return jnp.squeeze(jnp.clip(jnp.exp(log_r), min=1e-8, max=1e3))
        return jnp.squeeze(log_r)
