import os

# Personal: by default change the CUDA_VISIBLE_DEVICES to something less used on server
if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # TODO: remove before release

import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from typing import Sequence

from .encoder import encoder_factory
from .context import context_factory
from .rate_prediction import rate_prediction_factory
from .process_models import distribution_dictionary


class Embedding(nn.Module):
    encoder_args: dict
    context_args: dict
    latent_dim: int
    context_dim: int
    convolutional: Sequence[dict] = None

    def setup(
        self,
    ):
        self.encoder = encoder_factory(**self.encoder_args, latent_dim=self.latent_dim)
        self.context = context_factory(
            **self.context_args, context_dim=self.context_dim
        )
        # determine if context model is a RNN, execute scan if so
        if isinstance(self.context, nn.RNNCellBase):
            self.rnn_context = True
            self.context_scan = nn.RNN(self.context, time_major=False)
        else:
            self.rnn_context = False

        if self.convolutional is not None:
            raise ValueError("Convolutional context not stably implemented.")
            self.convolutional_layers = [
                nn.Conv(**layer, padding="SAME") for layer in self.convolutional
            ]
        else:
            self.convolutional_layers = []

    def __call__(self, x, delta_t):
        # attach delta_t to z to be used in context model
        z = jax.vmap(self.encoder, in_axes=(1), out_axes=1)(x)
        z_aug = jnp.concatenate([z, jnp.log(delta_t[..., None])], axis=-1)
        # generate context
        if self.rnn_context:
            # context is a RNN
            if self.context.infer_init:
                # use the context model to initialize the carry if implemented
                init_carry = self.context.initialize_carry_from_data(z_aug)
            else:
                init_carry = None
            c = self.context_scan(z_aug, initial_carry=init_carry)
        else:
            # context is an alternative model (e.g. Wavenet)
            # pass it the full sequence and let it handle the context
            c = self.context(z_aug)

        if len(self.convolutional_layers) > 0:
            # idea I was playing with prior to wavenet. will remove if wavenet continues to perform well
            raise ValueError("Convolutional context not stably implemented.")
            for layer in self.convolutional_layers[:-1]:
                c = layer(c)
                c = nn.leaky_relu(c)
            c = self.convolutional_layers[-1](c)
            print("convolutional", c.shape)

        return (
            z,
            c,
        )  # z = (n_batch, n_marks, latent_dim), c = (n_batch, n_marks, context_dim)


class C3PO(nn.Module):
    encoder_args: dict
    context_args: dict
    rate_args: dict
    distribution: str
    latent_dim: int
    context_dim: int
    n_neg_samples: int
    predicted_sequence_length: int = 1
    context_convolutional: Sequence[dict] = None

    def setup(self):
        self.embedding = Embedding(
            encoder_args=self.encoder_args,
            context_args=self.context_args,
            latent_dim=self.latent_dim,
            context_dim=self.context_dim,
            convolutional=self.context_convolutional,
        )
        self.distribution_class = distribution_dictionary[self.distribution]()
        self.rate_prediction = rate_prediction_factory(
            **self.rate_args,
            latent_dim=self.latent_dim,
            context_dim=self.context_dim,
            n_params=self.distribution_class.n_params
        )

    def _distribution_object(self):
        return distribution_dictionary[self.distribution]()

    def __call__(self, x, delta_t, rand_key):
        z, c = self.embedding(
            x, delta_t
        )  # z = (n_marks, latent_dim), c = (n_marks, context_dim)
        neg_z = get_neg_samples_batch(
            z, self.n_neg_samples, rand_key
        )  # (n_marks, n_neg_samples, latent_dim)

        vmap_params = jax.vmap(self.rate_prediction, in_axes=(0), out_axes=0)
        z_stacked = jnp.concat(
            [
                jnp.expand_dims(z[:, i : -self.predicted_sequence_length + i], axis=1)
                for i in range(self.predicted_sequence_length)
            ],
            axis=1,
        )  # (n_marks-predicted_sequence_length, predicted_sequence_length, latent_dim,
        print(z_stacked.shape)
        pos_params = jax.vmap(
            lambda zi: vmap_params(zi, c[:, : -self.predicted_sequence_length]),
            in_axes=1,
            out_axes=1,
        )(
            z_stacked
        )  # (n_marks-predicted_sequence_length, predicted_sequence_length, n_params)

        print(z.shape, pos_params.shape)
        print("neg_z", neg_z.shape)
        print("c", c.shape)
        neg_params = jax.vmap(
            lambda zi: vmap_params(zi, c[:, : -self.predicted_sequence_length]),
            in_axes=1,
            out_axes=1,
        )(neg_z[:, :, self.predicted_sequence_length :])

        print("neg_params", neg_params.shape)

        # cum_neg_rates = jnp.sum(neg_rates, axis=1)  # (n_marks)

        return pos_params, neg_params

    def embed(self, x, delta_t):
        return self.embedding(x, delta_t)

    # @jax.jit
    def loss_generalized_model(
        self,
        pos_parameters,
        neg_parameters,
        delta_t,
        sample_step=1,  # None,
        scale_neg_samples=30,
    ):
        """C3PO loss function for length n_predict and generic process.
        Parameters:
        pos_parameters: jnp.array of shape (batch_size, predicted_sequence_length, n_timepoints, n_params)
        neg_parameters: jnp.array of shape (batch_size, n_neg_samples, n_timepoints, n_params)
        delta_t: jnp.array of shape (batch_size, n_timepoints)
        sample_step: int, optional
            The step to sample the loss. The default is 1.
            Useful to prevent issues with long predicted sequences.
        scale_neg_samples: int, optional
            The scaling factor for the negative samples. The default is 10.
            This simulates increased number of negative samples without the memory cost.

        Returns:
            jnp.array: the loss value
        """
        if sample_step is None:
            sample_step = self.predicted_sequence_length

        delta_t_stacked = jnp.concatenate(
            [
                jnp.expand_dims(
                    delta_t[:, 1 + i : -self.predicted_sequence_length + i], axis=1
                )
                for i in range(self.predicted_sequence_length)
            ],
            axis=1,
        )
        cum_delta_t = jnp.cumsum(delta_t_stacked, axis=1)

        # print("cum_delta_t", cum_delta_t.shape, delta_t_stacked.shape)
        # print("pos_parameters", pos_parameters.shape)
        # hazard evaluation for when things fired
        hazard_term = self._distribution_object().log_hazard(
            cum_delta_t, pos_parameters[:, :, 1:]
        )

        # survival evaluation for neg samples after end of sequence
        neg_survival_term = self._distribution_object().log_survival(
            cum_delta_t[:, -1][:, None, :], neg_parameters[:, :, 1:]
        )
        pos_survival_term = self._distribution_object().log_survival(
            cum_delta_t[:, :-1], pos_parameters[:, 1:, 1:]
        )

        neg_log_p = -(
            jnp.sum(hazard_term, axis=1)[..., ::sample_step]
            + jnp.sum(neg_survival_term, axis=1)[..., ::sample_step]
            + jnp.sum(pos_survival_term, axis=1)[..., ::sample_step] * scale_neg_samples
        )
        return jnp.mean(neg_log_p)


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
