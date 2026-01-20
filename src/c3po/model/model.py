import os

# Personal: by default change the CUDA_VISIBLE_DEVICES to something less used on server
if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # TODO: remove before release

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import pmap
from flax.linen import Module
from functools import partial
import numpy as np
import optax
from typing import Sequence, Callable, Dict
from tqdm import tqdm

from .encoder import encoder_factory
from .context import context_factory
from .rate_prediction import rate_prediction_factory
from .process_models import distribution_dictionary
from .util import DilatedCausalConv1D, chunked_logsumexp
from .bidirectional_model import BidirectionalC3PO


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
        # print("z_stacked", z_stacked.shape)

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

        # print("Z", z.shape, "pos_params", pos_params.shape)
        # print("neg_z", neg_z.shape)
        # print("c", c.shape)
        # predict the rate parameters for the negative samples
        neg_params = jax.vmap(
            lambda zi: vmap_params(zi, c[:, : -self.predicted_sequence_length]),
            in_axes=1,
            out_axes=1,
        )(neg_z[:, :, self.predicted_sequence_length :])

        # print("neg_params", neg_params.shape)

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

    def mle_loss(
        self,
        pos_parameters,
        neg_parameters,
        delta_t,
        sample_step=1,  # None,
        n_emission_sources=20,
        rand_key=None,
        prior_params=None,
    ):
        """Maximum likelihood loss function for length n_predict and generic process.
        L = log(Hazard_pos) + log SUM_samples(Survival)

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
        if self.sample_params == "gaussian":
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

        if self.sample_params == "gaussian":
            # add the log probability of the samples
            neg_log_p += (jnp.sum(kl_div_pos) + jnp.sum(kl_div_neg)) * 10.0

        return jnp.mean(neg_log_p)

    def loss(self, pos_parameters, neg_parameters, delta_t, z, neg_z):
        """C3PO loss function for length n_predict and generic process.
        Parameters:
        pos_parameters: jnp.array of shape (batch_size, predicted_sequence_length, n_timepoints, n_params)
        neg_parameters: jnp.array of shape (batch_size, n_neg_samples, n_timepoints, n_params)
        delta_t: jnp.array of shape (batch_size, n_timepoints)
        rand_key: jnp.array, optional
            The random key to use for sampling. The default is None.

        Returns:
            jnp.array: the loss value
        """
        return self.contrastive_loss(
            self,
            pos_parameters,
            neg_parameters,
            delta_t,
            z,
            neg_z,
        )

    def contrastive_loss(
        self,
        pos_parameters,
        neg_parameters,
        delta_t,
        z,
        neg_z,
    ):
        """C3PO loss function for length n_predict and generic process.
        L = log[ (H_pos / H'_pos) / SUM_samples(H_i/H'_i) ]
        Where H is the learned hazard function and H' is the hazard function of the alternative (noise) model.
        (see docs for derivation)

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
            # alternative model
            # V1: assume intensity of noise falls off with L2 norm
            return -1 * jnp.sum(z**2, axis=-1)  # Gaussian hazard

        pos_log_noise_hazard_term = log_noise_hazard(z)[
            :, 1:
        ]  # (n_batch, n_timepoints, latent_dim)
        neg_log_noise_hazard_term = log_noise_hazard(neg_z)[:, :, 1:]
        neg_log_noise_hazard_term = jnp.concatenate(
            [neg_log_noise_hazard_term, pos_log_noise_hazard_term[:, None, :]], axis=1
        )

        neg_term = chunked_logsumexp(
            neg_log_hazard_term - neg_log_noise_hazard_term, axis=1, chunk_size=8
        )
        loss = -pos_log_hazard_term + pos_log_noise_hazard_term + neg_term
        return jnp.mean(loss)

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


# ----------------------------------------------------------------------------------
# Training functions


def _make_apply_fn(
    model: Module,
) -> Callable:
    """Create a pure, reusable apply fn (no closure over model at jit-time)."""

    def apply_fn(params: Dict, x, delta_t, rng):
        # Forward pass only depends on params & arrays.
        # If you need mutable collections, pass `mutable=...` here explicitly.
        return model.apply(params, x, delta_t, rng)

    return apply_fn


def _make_loss_fn(
    model: Module,
    loss_type: str = "contrastive",
    l1_penalty: float = None,
) -> Callable:
    """Return a jittable loss that does not close over changing Python objects."""
    apply_fn = _make_apply_fn(model)

    def loss_fn(params, x, delta_t, rng):
        pos_params, neg_params, z, _c, neg_z = apply_fn(params, x, delta_t, rng)
        if l1_penalty is not None:
            l1_loss = l1_penalty * jnp.sum(jnp.abs(_c), axis=-1).mean()
        else:
            l1_loss = 0.0

        if loss_type == "contrastive":
            # Keep this pure and array-driven; no Python branching on data.
            return (
                model.contrastive_loss(
                    pos_params,
                    neg_params,
                    delta_t,
                    z[:, model.predicted_sequence_length :],
                    neg_z,
                )
                + l1_loss
            )
        elif loss_type == "mle":
            return (
                model.mle_loss(pos_params, neg_params, delta_t, rand_key=rng) + l1_loss
            )
        else:
            raise ValueError("Unknown loss_type")

    return loss_fn


def train_model(
    model,
    params,
    x_train,
    delta_t_train,
    learning_rate=1e-3,
    n_epochs=1000,
    initial_batch_size=64,
    initial_n_neg=2,
    buffer_size=5,
    loss_type="contrastive",
    optimizer=None,
    max_n_neg=256,
    min_batch_size=4,
    multi_gpu=False,
    l1_penalty=None,
):
    """
    Train the C3PO model with adaptive negative sampling and batch size.

    Parameters:
        model (C3PO): The C3PO model to train.
        params (dict): The initial parameters of the model.
        x_train (jnp.array): The training data marks.
        delta_t_train (jnp.array): The training data time intervals.
        learning_rate (float): The learning rate for the optimizer.
        n_epochs (int): The maximum number of epochs to train.
        initial_batch_size (int): The initial batch size for training.
        initial_n_neg (int): The initial number of negative samples.
        buffer_size (int): The minimum number of epochs to wait before adjusting n_neg or batch size.
        loss_type (str): The type of loss function to use ('contrastive' or 'mle').
        optimizer (optax.GradientTransformation): The optimizer to use. If None, Adam is used.
        max_n_neg (int): The maximum number of negative samples.
        min_batch_size (int): The minimum batch size.
        multi_gpu (bool): Whether to use multiple GPUs for training.
        l1_penalty (float): L1 penalty on context embeddings. If None, no penalty is applied.

    Returns:
        params (dict): The trained model parameters.
        tracked_loss (list): The tracked loss values over epochs.
    """
    # Check inputs
    if np.min(delta_t_train) <= 0:
        raise ValueError("All delta_t values must be positive.")

    # prepare the optimizer and model for training
    if optimizer is None:
        optimizer = optax.chain(
            optax.adam(learning_rate),
        )
    opt_state = optimizer.init(params)
    model = update_n_neg(model, initial_n_neg)

    tracked_loss = []
    batch_size = initial_batch_size
    n_neg = initial_n_neg
    buffer = buffer_size

    # create jittable loss and grad functions
    loss_fn = _make_loss_fn(model, loss_type=loss_type, l1_penalty=l1_penalty)
    if not multi_gpu:
        loss_grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    else:
        # loss_grad_fn = pmap(loss_grad_fn, in_axes=(None, 0, 0, 0))
        train_step_fn = make_multi_gpu_train_step(loss_fn, optimizer)

    try:
        for i in range(n_epochs):
            # apply one epoch of training
            if multi_gpu:
                # raise NotImplementedError("Multi-GPU training not implemented yet.")
                params, opt_state, epoch_loss = train_epoch_multi_gpu(
                    model,
                    params,
                    x_train,
                    delta_t_train,
                    jax.random.PRNGKey(i),
                    batch_size,
                    optimizer,
                    opt_state,
                    # loss_grad_fn,
                    train_step_fn,
                    epoch_number=i + 1,
                    n_neg=n_neg,
                )
            else:
                params, opt_state, epoch_loss = train_epoch(
                    model,
                    params,
                    x_train,
                    delta_t_train,
                    jax.random.PRNGKey(i),
                    batch_size,
                    optimizer,
                    opt_state,
                    loss_grad_fn,
                    epoch_number=i + 1,
                    n_neg=n_neg,
                )
            tracked_loss.append(epoch_loss)
            buffer -= 1

            # allow at least buffer_size epochs before adjusting
            if buffer > 0:
                continue

            # check if stalled
            if np.mean(tracked_loss[-5:-1]) >= tracked_loss[-1] * 1.01:
                continue  # still improving

            # stalled, try to increase n_neg
            if n_neg < max_n_neg:
                # update n_neg, model, and loss functions
                n_neg = min(n_neg * 2, max_n_neg)
                model = update_n_neg(model, n_neg)
                # update the training functions
                loss_fn = _make_loss_fn(
                    model, loss_type=loss_type, l1_penalty=l1_penalty
                )
                if not multi_gpu:
                    del loss_grad_fn  # free memory
                    jax.clear_caches()
                    loss_grad_fn = jax.jit(jax.value_and_grad(loss_fn))
                else:
                    del train_step_fn  # free memory
                    jax.clear_caches()
                    train_step_fn = make_multi_gpu_train_step(loss_fn, optimizer)
                buffer = buffer_size  # reset buffer
                continue

            # stalled, try to adjust batch size
            elif batch_size > min_batch_size:
                # update batch_size, n_neg, model, and loss functions
                batch_size = max(batch_size // 2, min_batch_size)
                buffer = buffer_size  # reset buffer
                n_neg = max(initial_n_neg, max_n_neg // 8)
                model = update_n_neg(model, n_neg)
                # update the training functions
                loss_fn = _make_loss_fn(
                    model, loss_type=loss_type, l1_penalty=l1_penalty
                )
                if not multi_gpu:
                    del loss_grad_fn  # free memory
                    jax.clear_caches()
                    loss_grad_fn = jax.jit(jax.value_and_grad(loss_fn))
                else:
                    del train_step_fn  # free memory
                    jax.clear_caches()
                    # loss_grad_fn = pmap(loss_grad_fn, in_axes=(None, 0, 0, 0))
                    train_step_fn = make_multi_gpu_train_step(loss_fn, optimizer)
                buffer = buffer_size  # reset buffer
                print(
                    f"Stalled training, decreasing batch size to {batch_size} and n_neg to {n_neg}"
                )
                buffer = buffer_size  # reset buffer
                continue

            else:
                print("Stalled training, no further adjustments possible.")
                break
    except KeyboardInterrupt:
        print("Training interrupted by user.")

    return params, tracked_loss


def train_epoch(
    model,
    params,
    x_train,
    delta_t_train,
    rand_key,
    batch_size,
    optimizer,
    opt_state,
    loss_grad_fn,
    epoch_number=None,
    n_neg=None,
):
    """Train the model for one epoch.

    Args:
        model (C3PO): The C3PO model to train.
        params (dict): The model parameters.
        x_train (jnp.array): The training data marks.
        delta_t_train (jnp.array): The training data time intervals.
        rand_key (jnp.array): The random key for JAX.
        batch_size (int): The batch size for training.
        optimizer (optax.GradientTransformation): The optimizer to use.
        opt_state (optax.OptState): The optimizer state.
        loss_grad_fn (Callable): The loss and gradient function.
        epoch_number (int, optional): The epoch number for logging. Defaults to None.
        n_neg (int, optional): The number of negative samples for logging. Defaults to None.

    Raises:
        ValueError: If the loss is not finite.
    Returns:
        params (dict): The updated model parameters.
        opt_state (optax.OptState): The updated optimizer state.
        average_epoch_loss (float): The average loss for the epoch.
    """
    ind = np.arange(x_train.shape[0])
    np.random.shuffle(ind)
    epoch_loss = []
    j = 0
    with tqdm(
        total=x_train.shape[0], desc=f"Epoch {epoch_number}", unit="samples"
    ) as pbar:
        prev_params = params.copy()  # store params from end of previous epoch
        while j < x_train.shape[0]:
            # Perform one gradient update.
            rand_key, _ = jax.random.split(rand_key)
            batch_inds = ind[j : j + batch_size]
            if len(batch_inds) < batch_size:
                break
            loss_val, grads = loss_grad_fn(
                params, x_train[batch_inds], delta_t_train[batch_inds], rand_key
            )
            if not np.isfinite(loss_val):
                params = prev_params.copy()
                raise ValueError("Loss is not finite")
            epoch_loss.append(loss_val)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            j += batch_size

            pbar.update(batch_size)
            pbar.set_postfix(
                loss=np.mean(epoch_loss), n_neg=n_neg, batch_size=batch_size
            )

    average_epoch_loss = np.mean(epoch_loss)
    return params, opt_state, average_epoch_loss


def make_multi_gpu_train_step(loss_fn, optimizer):
    """Create a function to perform a training step using multiple GPUs.

    Args:
        loss_fn (Callable): The loss function to use.
        optimizer (optax.GradientTransformation): The optimizer to use.

    Returns:
        Callable: The multi-GPU train step function. Inputs should be split across devices.
            Args:
                params (dict): The model parameters.
                opt_state (optax.OptState): The optimizer state.
                x (jnp.array): The input data marks.
                delta_t (jnp.array): The input data time intervals.
                rng (jnp.array): The random key for JAX.
            Returns:
                params (dict): The updated model parameters.
                opt_state (optax.OptState): The updated optimizer state.
                loss (float): The loss value.
    """

    @partial(pmap, axis_name="devices")
    def train_step(params, opt_state, x, delta_t, rng):
        # rand_split = jax.random.split(rng, jax.local_device_count())
        # value_and_grad = jax.pmap(jax.value_and_grad(loss_fn), axis_name="devices")
        value_and_grad = jax.value_and_grad(loss_fn)
        loss, grads = value_and_grad(params, x, delta_t, rng)

        # Average across devices
        loss = jax.lax.pmean(loss, axis_name="devices")
        grads = jax.lax.pmean(grads, axis_name="devices")

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return train_step


def train_epoch_multi_gpu(
    model,
    params,
    x_train,
    delta_t_train,
    rand_key,
    batch_size,
    optimizer,
    opt_state,
    # loss_grad_fn,
    train_step_fn,
    epoch_number=None,
    n_neg=None,
):
    """Train the model for one epoch using multiple GPUs.

    Args:
        model (C3PO): The C3PO model to train.
        params (dict): The model parameters.
        x_train (jnp.array): The training data marks.
        delta_t_train (jnp.array): The training data time intervals.
        rand_key (jnp.array): The random key for JAX.
        batch_size (int): The batch size for training.
        optimizer (optax.GradientTransformation): The optimizer to use.
        opt_state (optax.OptState): The optimizer state.
        train_step_fn (Callable): The multi-GPU train step function.
        epoch_number (int, optional): The epoch number for logging. Defaults to None.
        n_neg (int, optional): The number of negative samples for logging. Defaults to None
    Returns::
        params (dict): The updated model parameters.
        opt_state (optax.OptState): The updated optimizer state.
        average_epoch_loss (float): The average loss for the epoch.
    """
    split_params = jax.device_put_replicated(params, jax.devices())
    split_opt_state = jax.device_put_replicated(opt_state, jax.devices())
    # Number of available GPUs
    num_devices = jax.device_count()
    total_batch_size = batch_size * num_devices

    def get_batches(x, delta_t):
        """Simulate getting independent batches for each device."""
        return np.array([x[i::num_devices] for i in range(num_devices)]), np.array(
            [delta_t[i::num_devices] for i in range(num_devices)]
        )

    ind = np.arange(x_train.shape[0])
    np.random.shuffle(ind)
    epoch_loss = []
    j = 0
    with tqdm(
        total=x_train.shape[0], desc=f"Epoch {epoch_number}", unit="samples"
    ) as pbar:
        while j < x_train.shape[0]:
            # Perform one gradient update.
            rand_key, _ = jax.random.split(rand_key)
            batch_inds = ind[j : j + total_batch_size]
            if batch_inds.shape[0] < total_batch_size:
                break

            x_batch, delta_t_batch = get_batches(
                x_train[batch_inds],
                delta_t_train[batch_inds],
            )
            # print(x_batch.shape, delta_t_batch.shape)
            # loss_val, grads = loss_grad_fn(
            #     split_params, x_batch, delta_t_batch, rand_key
            # )

            # if not np.isfinite(loss_val):
            #     raise ValueError("Loss is not finite")
            # epoch_loss.append(loss_val)
            # updates, opt_state = optimizer.update(grads, opt_state, params)
            # params = optax.apply_updates(params, updates)

            split_params, split_opt_state, loss_val = train_step_fn(
                split_params,
                split_opt_state,
                x_batch,
                delta_t_batch,
                jax.random.split(rand_key, num_devices),
            )

            if not np.isfinite(loss_val.mean()):
                raise ValueError("Loss is not finite")

            epoch_loss.append(loss_val.mean())

            j += total_batch_size

            pbar.update(total_batch_size)
            pbar.set_postfix(
                loss=np.mean(epoch_loss), n_neg=n_neg, batch_size=batch_size
            )

    merged_params = jax.tree_util.tree_map(lambda x: x[0], split_params)
    average_epoch_loss = np.mean(epoch_loss)
    opt_state = jax.tree_util.tree_map(lambda x: x[0], split_opt_state)

    return merged_params, opt_state, average_epoch_loss


def update_n_neg(model, new_n_neg):
    """Updates the number of negative samples in the model.

    Args:
        model (C3PO): The model to update.
        new_n_neg (int): The new number of negative samples.

    Returns:
        C3PO: The updated model.
    """
    model_class = BidirectionalC3PO if isinstance(model, BidirectionalC3PO) else C3PO

    new_model = model_class(
        encoder_args=model.encoder_args,
        context_args=model.context_args,
        rate_args=model.rate_args,
        distribution=model.distribution,
        latent_dim=model.latent_dim,
        context_dim=model.context_dim,
        n_neg_samples=new_n_neg,
        predicted_sequence_length=model.predicted_sequence_length,
        context_convolutional=model.context_convolutional,
        sample_params=model.sample_params,
        return_embeddings_in_call=model.return_embeddings_in_call,
    )
    return new_model
