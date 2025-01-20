import os

# Personal: by default change the CUDA_VISIBLE_DEVICES to something less used on server
if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # TODO: remove before release

from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import scan
from functools import partial

from .encoder import encoder_factory
from .context import context_factory
from .rate_prediction import rate_prediction_factory


class Embedding(nn.Module):
    encoder_args: dict
    context_args: dict
    latent_dim: int
    context_dim: int

    def setup(self):
        self.encoder = encoder_factory(**self.encoder_args, latent_dim=self.latent_dim)
        self.context = context_factory(
            **self.context_args, context_dim=self.context_dim
        )
        self.init_carry = self.param(
            "carry", nn.initializers.lecun_normal(), (self.context_dim, 1)
        )

        self.context_scan = nn.RNN(self.context, time_major=False)

    def __call__(self, x, delta_t):
        z = jax.vmap(self.encoder, in_axes=(1), out_axes=1)(x)
        z_aug = z
        # z_aug = jnp.concatenate([z, jnp.log(delta_t[..., None])], axis=-1)

        print("init carry", self.init_carry.shape)
        c_0 = jnp.ones((x.shape[0], self.context_dim)) * self.init_carry[None, :, 0]
        c_0 = (c_0, jnp.zeros((x.shape[0], 1)))
        # c_0 = c_0 + jax.random.normal(0, (x.shape[0], self.context_dim)) * 0.1
        c_0 = None
        c = self.context_scan(z_aug, initial_carry=c_0)
        return (
            z,
            c,
        )  # z = (n_batch, n_marks, latent_dim), c = (n_batch, n_marks, context_dim)


class LFADSEmbedding(nn.Module):
    encoder_args: dict
    context_args: dict
    latent_dim: int
    context_dim: int

    def setup(self):
        self.encoder = encoder_factory(**self.encoder_args, latent_dim=self.latent_dim)
        self.context_forward = context_factory(
            **self.context_args["forward"], context_dim=self.context_dim
        )
        self.context_reverse = context_factory(
            **self.context_args["reverse"], context_dim=self.context_dim
        )
        self.lfads = context_factory(
            **self.context_args["lfads"], context_dim=self.context_dim
        )

        self.init_carry = self.variable(
            "state", "carry", lambda: jnp.zeros((self.context_dim,))
        )
        self.context_scan_forward = nn.RNN(self.context_forward, time_major=False)
        self.context_scan_reverse = nn.RNN(
            self.context_reverse, time_major=False, keep_order=True, reverse=True
        )

        self.lfads_scan = nn.RNN(self.lfads, time_major=False, return_carry=True)

    def __call__(self, x, delta_t):
        z = jax.vmap(self.encoder, in_axes=(1), out_axes=1)(x)
        z_aug = jnp.concatenate([z, jnp.log(delta_t[..., None])], axis=-1)

        c_forward = self.context_scan_forward(z_aug)
        c_reverse = self.context_scan_reverse(
            z_aug,
        )
        lfads_init = jnp.tanh(c_forward[:, -1, :] + c_reverse[:, -1, :])
        print("lfads_init", lfads_init.shape)
        print("delta_t", delta_t.shape)
        _, c = self.lfads_scan(delta_t[:, :, None], initial_carry=lfads_init)
        print("c", c.shape)
        print("z", z.shape)
        print("_", _.shape)
        return (
            z,
            c,
        )  # z = (n_batch, n_marks, latent_dim), c = (n_batch, n_marks, context_dim)


class C3PO(nn.Module):
    encoder_args: dict
    context_args: dict
    rate_args: dict
    latent_dim: int
    context_dim: int
    n_neg_samples: int
    lfads: bool = False
    initial_carry_args: dict = None

    def setup(self):
        if self.lfads:
            self.embedding = LFADSEmbedding(
                encoder_args=self.encoder_args,
                context_args=self.context_args,
                latent_dim=self.latent_dim,
                context_dim=self.context_dim,
            )
        else:
            self.embedding = Embedding(
                encoder_args=self.encoder_args,
                context_args=self.context_args,
                latent_dim=self.latent_dim,
                context_dim=self.context_dim,
            )
        self.rate_prediction = rate_prediction_factory(
            **self.rate_args, latent_dim=self.latent_dim, context_dim=self.context_dim
        )

    def __call__(self, x, delta_t, rand_key):
        z, c = self.embedding(
            x, delta_t
        )  # z = (n_marks, latent_dim), c = (n_marks, context_dim)
        neg_z = get_neg_samples_batch(
            z, self.n_neg_samples, rand_key
        )  # (n_marks, n_neg_samples, latent_dim)

        vmap_rates = jax.vmap(self.rate_prediction, in_axes=(0), out_axes=0)
        pos_rates = vmap_rates(z[:, 1:], c[:, :-1])  # (n_marks)

        print(z.shape, pos_rates.shape)
        print("neg_z", neg_z.shape)
        print("c", c.shape)
        neg_rates = jax.vmap(
            lambda zi: vmap_rates(zi, c[:, :-1]), in_axes=1, out_axes=1
        )(neg_z[:, :, 1:])

        print("neg_rates", neg_rates.shape)

        cum_neg_rates = jnp.sum(neg_rates, axis=1)  # (n_marks)

        return pos_rates, cum_neg_rates

    def embed(self, x, delta_t):
        return self.embedding(x, delta_t)


def get_neg_samples_batch(z, n_neg_samples, rand_key):
    """Get negative samples for full batch.

    Args:
        z jnp.array: the embedded marks (n_marks, latent_dim)
        n_neg_samples int: the number of negative samples to draw
        rand_key jnp.array: the random key

    Returns:
        _type_: _description_
    """
    neg_sampler = jax.jit(partial(get_neg_samples, z, n_neg_samples))
    return jax.vmap(neg_sampler)(jax.random.split(rand_key, z.shape[0]))


def get_neg_samples(z, n_neg_samples, rand_key):
    neg_sampler = jax.jit(partial(neg_sample, z))
    return jax.vmap(neg_sampler)(jax.random.split(rand_key, n_neg_samples))


def neg_sample(z, rand_key):
    batch_samples = jax.random.choice(rand_key, z.shape[0], shape=(z.shape[1],))
    time_samples = jax.random.choice(rand_key, z.shape[1], shape=(z.shape[1],))
    return z[batch_samples, time_samples]


def loss_sample(pos_rates, cum_neg_rates, delta_t):
    neg_log_p = jnp.log(pos_rates**-1) + delta_t * (
        cum_neg_rates + pos_rates
    )  # (n_marks)
    return jnp.mean(neg_log_p)


@jax.jit
def loss(pos_rates, cum_neg_rates, delta_t):
    """C3PO loss function for length one sequence and poisson process."""
    neg_log_p = -jnp.log(pos_rates) + 1 / 3 * delta_t[:, 1:] * (
        cum_neg_rates + pos_rates
    )  # (n_marks)
    return jnp.mean(neg_log_p)