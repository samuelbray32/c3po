import os

# Personal: by default change the CUDA_VISIBLE_DEVICES to something less used on server
if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # TODO: remove before release

import jax
import jax.numpy as jnp

# import flax.linen as nn
# from jax import pmap
# from flax.linen import Module
# from functools import partial
# import numpy as np
# import optax
from typing import Sequence, Callable, Dict
from tqdm import tqdm

from .encoder import encoder_factory
from .context import context_factory
from .rate_prediction import rate_prediction_factory
from .process_models import distribution_dictionary
from .util import DilatedCausalConv1D, chunked_logsumexp

from .model import C3PO, get_neg_samples_batch


class BidirectionalC3PO(C3PO):

    def setup(self):
        if not self.rate_args.get("rate_model", None) == "sharedSpace":
            raise ValueError(
                "BidirectionalC3PO only supports 'sharedSpace' rate model."
            )

        # Set up context model as bidirectional C3PO
        if not self.context_args.get("context_model", None) == "bidirectional_c3po":
            raise ValueError(
                "BidirectionalC3PO requires context_model to be 'bidirectional_c3po'."
                + "/n To convert your args please follow the example below: \n"
                + "context_args = dict( \n"
                + '    context_model="bidirectional_c3po", \n'
                + "    forward_model_args=your_previous_context_args, \n"
                + "    backward_model_args=your_previous_context_args, \n"
                + ")"
            )

        if not self.distribution == "poisson":
            raise ValueError("BidirectionalC3PO only supports 'poisson' distribution.")

        super().setup()

    def __call__(self, x, delta_t, rand_key):
        # Embed the marks and get the context
        if self.embedding.requires_random_key:
            z, c = self.embedding(x, delta_t, rand_key)
        else:
            z, c = self.embedding(
                x, delta_t
            )  # z = (n_marks, latent_dim), c = (n_marks, context_dim)

        # parse the forward and backward context
        c_forward, c_backward = c

        neg_c_backward = get_neg_samples_batch(
            c_backward, self.n_neg_samples, rand_key
        )  # (n_marks, n_neg_samples, context_dim)

        # predict the rate parameters for the the observed sequences
        # rates of sequences following time i (z_stacked[n_batch,i]) are predicted
        # from context at time i (c[n_batch,i])
        pos_params = jnp.sum(
            c_forward[:, :-1] * c_backward[:, 1:], axis=-1, keepdims=True
        )  # (n_marks-predicted_sequence_length, 1)

        # predict the rate parameters for the negative samples
        vmap_neg_params = jax.vmap(
            lambda zi: jnp.sum(c_forward[:, :-1] * zi[:, 1:], axis=-1, keepdims=True),
            in_axes=1,
            out_axes=1,
        )
        neg_params = vmap_neg_params(neg_c_backward)  # (n_marks, n_neg_samples, 1)

        if self.return_embeddings_in_call:
            return (
                pos_params,
                neg_params,
                z[:, :],
                c,
                neg_c_backward,
            )
        return pos_params, neg_params

    def loss(self, pos_parameters, neg_parameters, *args, **kwargs):
        return self.contrastive_loss(pos_parameters, neg_parameters)

    def contrastive_loss(self, pos_parameters, neg_parameters, *args, **kwargs):
        expanded_neg_params = jnp.concatenate(
            [
                neg_parameters,
                pos_parameters[
                    :,
                    None,
                ],
            ],
            axis=1,
        )
        neg_term = chunked_logsumexp(expanded_neg_params, axis=1, chunk_size=8)
        print("LOSS:", neg_term.shape, pos_parameters.shape)
        loss = -pos_parameters + neg_term
        return jnp.mean(loss)
