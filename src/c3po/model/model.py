import os

# Personal: by default change the CUDA_VISIBLE_DEVICES to something less used on server
if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # TODO: remove before release

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.scipy.special import logsumexp
from functools import partial
from typing import Sequence

from .encoder import encoder_factory
from .context import context_factory
from .rate_prediction import rate_prediction_factory
from .process_models import distribution_dictionary
from .util import DilatedCausalConv1D, chunked_logsumexp


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
            self.convolutional_layers = [
                DilatedCausalConv1D(**layer) for layer in self.convolutional
            ]
        else:
            self.convolutional_layers = []

    def __call__(self, x, delta_t, rand_key=None):
        # attach delta_t to z to be used in context model
        if self.encoder.requires_random_key:
            if rand_key is None:
                raise ValueError("Random key is required for this encoder.")
            z = jax.vmap(self.encoder, in_axes=(1, None), out_axes=1)(x, rand_key)
        else:
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
            for layer in self.convolutional_layers[:-1]:
                c = layer(c)
                c = nn.leaky_relu(c)
            c = self.convolutional_layers[-1](c)
            print("convolutional", c.shape)

        return (
            z,
            c,
        )  # z = (n_batch, n_marks, latent_dim), c = (n_batch, n_marks, context_dim)

    @property
    def requires_random_key(self):
        return self.encoder.requires_random_key


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
    sample_params: str = None
    return_embeddings_in_call: bool = False
    """
    C3PO model for continuous time point processes.
    Parameters:
    encoder_args: dict
        Arguments for the encoder model
    context_args: dict
        Arguments for the context model
    rate_args: dict
        Arguments for the rate prediction model
    distribution: str
        The distribution of the spiking process
    latent_dim: int
        The dimension of the latent space (Z)
    context_dim: int
        The dimension of the context space (C)
    n_neg_samples: int
        The number of negative samples to draw
    predicted_sequence_length: int, optional
        The length of the predicted sequence. The default is 1.
    context_convolutional: Sequence[dict], optional
        The convolutional layers post model. The default is None. Not stably implemented.
    sample_params: str, optional
        The method to sample the parameters. The default is None. Valid options are None, "gaussian".

    """

    def setup(self):
        self.embedding = Embedding(
            encoder_args=self.encoder_args,
            context_args=self.context_args,
            latent_dim=self.latent_dim,
            context_dim=self.context_dim,
            convolutional=self.context_convolutional,
        )
        self.distribution_class = distribution_dictionary[self.distribution]()
        n_params = self.distribution_class.n_params
        if self.sample_params == "gaussian":
            n_params = 2 * n_params
        self.rate_prediction = rate_prediction_factory(
            **self.rate_args,
            latent_dim=self.latent_dim,
            context_dim=self.context_dim,
            n_params=n_params,
        )

    def _distribution_object(self):
        return distribution_dictionary[self.distribution]()

    def __call__(self, x, delta_t, rand_key):
        # Embed the marks and get the context
        if self.embedding.requires_random_key:
            z, c = self.embedding(x, delta_t, rand_key)
        else:
            z, c = self.embedding(
                x, delta_t
            )  # z = (n_marks, latent_dim), c = (n_marks, context_dim)
        neg_z = get_neg_samples_batch(
            z, self.n_neg_samples, rand_key
        )  # (n_marks, n_neg_samples, latent_dim)

        # Stack sequences for prediction. sample z_stacked[n_batch,i] corresponds to the
        # sequence following the i-th mark in the batch.
        # This is done to allow the rate prediction model to predict the rates for the next
        # predicted_sequence_length marks after each mark in the batch.
        z_stacked = jnp.concat(
            [
                jnp.expand_dims(
                    z[:, 1 + i : z.shape[1] - self.predicted_sequence_length + i + 1],
                    axis=1,
                )
                for i in range(self.predicted_sequence_length)
            ],
            axis=1,
        )  # (n_marks-predicted_sequence_length, predicted_sequence_length, latent_dim,
        print("z_stacked", z_stacked.shape)

        # predict the rate parameters for the the observed sequences
        # rates of sequences following time i (z_stacked[n_batch,i]) are predicted
        # from context at time i (c[n_batch,i])
        vmap_params = jax.vmap(self.rate_prediction, in_axes=(0), out_axes=0)
        pos_params = jax.vmap(
            lambda zi: vmap_params(zi, c[:, : -self.predicted_sequence_length]),
            in_axes=1,
            out_axes=1,
        )(
            z_stacked
        )  # (n_marks-predicted_sequence_length, predicted_sequence_length, n_params)

        print("Z", z.shape, "pos_params", pos_params.shape)
        print("neg_z", neg_z.shape)
        print("c", c.shape)
        # predict the rate parameters for the negative samples
        neg_params = jax.vmap(
            lambda zi: vmap_params(zi, c[:, : -self.predicted_sequence_length]),
            in_axes=1,
            out_axes=1,
        )(neg_z[:, :, self.predicted_sequence_length :])

        print("neg_params", neg_params.shape)

        # cum_neg_rates = jnp.sum(neg_rates, axis=1)  # (n_marks)

        if self.return_embeddings_in_call:
            return (
                pos_params,
                neg_params,
                z[:, :],
                c[:, :],
                neg_z[:, :, self.predicted_sequence_length :],
            )
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
        n_emission_sources=20,
        rand_key=None,
        prior_params=None,
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

        # sample parameters if needed
        if self.sample_params is "gaussian":
            n_sigma = pos_parameters.shape[-1] // 2
            sigma_regularization = jnp.mean(
                jnp.log1p(jnp.exp(pos_parameters[..., n_sigma:]))
            ) + jnp.mean(
                jnp.log1p(jnp.exp(neg_parameters[..., n_sigma:]))
            )  # average variance
            sigma_regularization = 1 / sigma_regularization
            pos_parameters, log_pos_sample_prob = self.gauss_sample(
                rand_key, pos_parameters
            )
            neg_parameters, log_neg_sample_prob = self.gauss_sample(
                rand_key, neg_parameters
            )
            mu_pos = pos_parameters[..., :n_sigma]
            mu_neg = neg_parameters[..., :n_sigma]
            sigma_pos = jnp.log1p(jnp.exp(pos_parameters[..., n_sigma:]))
            sigma_neg = jnp.log1p(jnp.exp(neg_parameters[..., n_sigma:]))
            kl_div_pos = jnp.log(prior_params["sigma"] / sigma_pos) + (
                (sigma_pos**2 + (mu_pos - prior_params["mu"]) ** 2)
                / (2 * prior_params["sigma"] ** 2)
                - 0.5
            )
            kl_div_neg = jnp.log(prior_params["sigma"] / sigma_neg) + (
                (sigma_neg**2 + (mu_neg - prior_params["mu"]) ** 2)
                / (2 * prior_params["sigma"] ** 2)
                - 0.5
            )

        # hazard evaluation for when things fired
        hazard_term = self._distribution_object().log_hazard(
            cum_delta_t, pos_parameters[:, :, 1:]
        )

        # survival evaluation for neg samples after end of sequence
        neg_survival_term = self._distribution_object().log_survival(
            cum_delta_t[:, -1][:, None, :], neg_parameters[:, :, 1:]
        )
        neg_survival_term = jnp.mean(neg_survival_term, axis=1) * n_emission_sources

        pos_survival_term = self._distribution_object().log_survival(
            cum_delta_t[:, :-1], pos_parameters[:, 1:, 1:]
        )

        print("neg_survival_term", neg_survival_term.shape)
        print("pos_survival_term", pos_survival_term.shape)
        print("hazard_term", hazard_term.shape)

        neg_log_p = -(
            jnp.sum(hazard_term, axis=1)[..., ::sample_step]
            + neg_survival_term[..., ::sample_step]
            + jnp.sum(pos_survival_term, axis=1)[..., ::sample_step]
        )  # TODO: scale neg samples ?

        if self.sample_params is "gaussian":
            # add the log probability of the samples
            neg_log_p += (jnp.sum(kl_div_pos) + jnp.sum(kl_div_neg)) * 10.0

        return jnp.mean(neg_log_p)

    def contrastive_sequence_loss(
        self,
        pos_parameters,
        neg_parameters,
        delta_t,
        z,
        neg_z,
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

        # hazard evaluation for when things fired
        pos_log_hazard_term = self._distribution_object().log_hazard(
            cum_delta_t, pos_parameters[:, :, 1:]
        )

        neg_log_hazard_term = self._distribution_object().log_hazard(
            cum_delta_t[:, -1][:, None, :], neg_parameters[:, :, 1:]
        )
        neg_log_hazard_term = jnp.concatenate(
            [neg_log_hazard_term, pos_log_hazard_term], axis=1
        )

        def log_noise_hazard(z):
            # V1: assume intensity of noise falls off with L2 norm
            return -1 * jnp.sum(z**2, axis=-1)  # Gaussian hazard

        pos_log_noise_hazard_term = log_noise_hazard(z)[
            :, 1:
        ]  # (n_batch, n_timepoints, latent_dim)
        neg_log_noise_hazard_term = log_noise_hazard(neg_z)[:, :, 1:]
        neg_log_noise_hazard_term = jnp.concatenate(
            [neg_log_noise_hazard_term, pos_log_noise_hazard_term[:, None, :]], axis=1
        )

        # neg_term = logsumexp(neg_log_hazard_term - neg_log_noise_hazard_term, axis=1)
        neg_term = chunked_logsumexp(
            neg_log_hazard_term - neg_log_noise_hazard_term, axis=1, chunk_size=8
        )
        loss = -pos_log_hazard_term + pos_log_noise_hazard_term + neg_term
        return jnp.mean(loss)

    # def noise_hazard(self, z):
    #     # V1: assume intensity of noise falls off with distance
    #     return ((2 * 3.14) ** (self.latent_dim / 2)) * jnp.exp(
    #         -jnp.dot(
    #             z,
    #             z,
    #         )
    #         / 2
    #     )  # Gaussian hazard

    # def log_noise_hazard(self, z):
    #     # V1: assume intensity of noise falls off with distance
    #     return (self.latent_dim / 2) * jnp.log(2 * 3.14) - jnp.dot(
    #         z,
    #         z,
    #     ) / 2  # Gaussian hazard

    @staticmethod
    def gauss_sample(key, params):
        n_params = params.shape[-1] // 2
        mu = params[..., :n_params]
        sigma = jnp.log1p(jnp.exp(params[..., n_params:]))
        sample_vals = jax.random.normal(key, mu.shape)
        log_sample_prob = -0.5 * jnp.log(2 * jnp.pi) - sample_vals**2 / 2
        return sample_vals * sigma + mu, log_sample_prob


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
