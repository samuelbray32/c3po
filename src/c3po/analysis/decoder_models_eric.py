from __future__ import annotations

from typing import TYPE_CHECKING
from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from non_local_detector import Environment
try:
    from non_local_detector import Environment
except ImportError:
    Environment = None

from .decoder_models import (
    GraphDiscretizer,
    get_distances_to_interior_bins,
    graph_distance_between_points,
)


class EricBasedDecoder:
    def __init__(
        self,
        environment: Environment,
        predict_method: str = "weighted",
        kernel_args: dict = {"method": "gaussian", "sigma": 1.0},
        feature_smooth_std: float = None,
    ):
        self.environment = environment
        self.predict_method = predict_method
        self.discretizer = GraphDiscretizer(environment)
        self.prediction_func = None
        self.feature_smooth_std = feature_smooth_std
        self.kernel_args = kernel_args

        self.multidim = False

    def fit(self, X, y):
        ind = np.logical_and(~np.isnan(y).any(axis=1), np.all(~np.isnan(X), axis=1))
        X = X[ind]
        y = y[ind]
        y_fit = y.reshape(-1, 1) if not self.multidim else y
        y_binned = self.discretizer.fit_transform(y.reshape(-1, 1))

        if self.feature_smooth_std is not None:
            bin_ids, smoothed_weights = self.discretizer.get_smoothed_labels(
                y_fit, position_std=self.feature_smooth_std
            )
            row_sums = np.nansum(smoothed_weights, axis=1, keepdims=True)
            smoothed_weights = smoothed_weights / row_sums
            ind_valid = ~np.any(np.isnan(smoothed_weights), axis=1)
            # smoothed_weights = smoothed_weights[ind_valid]
            X = X[ind_valid]
            y_binned = smoothed_weights[ind_valid]
        if y_binned.ndim == 1:
            y_binned = np.eye(self.n_bins)[y_binned.astype(int)]

        # self.x_fit = X
        # self.y_fit = y_binned
        self.x_fit = jax.device_put(jnp.asarray(X, dtype=jnp.float32))
        self.y_fit = jax.device_put(jnp.asarray(y_binned, dtype=jnp.float32))
        return

    def _observed_bin_centers(self):
        interior_bin_ids = self.discretizer.interior_bin_ids()
        return self.discretizer.bin_centers[interior_bin_ids]

    def predict(self, X, return_posterior=False, chunk_size=1000, subsample=None):
        method_args = self.kernel_args.copy()
        method = method_args.pop("method")

        if self.prediction_func is None:
            # @jax.jit
            # def predict_single_posterior(x):
            #     score = SCORE_METHODS[method](x, self.x_fit, **method_args)
            #     return  jnp.sum(score[:, None] * self.y_fit, axis=0)

            # @jax.jit
            # def predict_batch_posterior(x):
            #     return jax.vmap(predict_single_posterior)(x)

            # self.prediction_func = predict_batch_posterior
            self.prediction_func = SCORE_METHODS[method](
                self.x_fit[::subsample], self.y_fit[::subsample], **method_args
            )

        # posterior = self.prediction_func(X)
        posterior_chunks = []

        for start in tqdm(
            range(0, len(X), chunk_size),
            total=len(X) // chunk_size + 1,
            desc="Predicting posterior",
        ):
            stop = min(start + chunk_size, len(X))
            posterior_chunk = self.prediction_func(X[start:stop])

            # Move each completed chunk back to host memory.
            posterior_chunks.append(np.asarray(posterior_chunk))

        posterior = np.concatenate(
            posterior_chunks,
            axis=0,
        )

        bin_centers = self._observed_bin_centers()
        if self.predict_method == "weighted":
            y_pred = np.dot(posterior, bin_centers)
        elif self.predict_method == "peak":
            y_pred = bin_centers[np.argmax(posterior, axis=1)]
        elif self.predict_method == "weighted_angle":
            complex_bin_centers = np.exp(1j * bin_centers)
            y_pred_complex = np.dot(posterior, complex_bin_centers)
            y_pred = np.angle(y_pred_complex)
        else:
            raise ValueError(
                f"Unknown predict_method: {self.predict_method}. Choose 'weighted', 'peak', or 'weighted_angle'."
            )
        if return_posterior:
            return y_pred[:, None], (posterior, self._observed_bin_centers())
        return y_pred[:, None]


# @jax.jit
# def gaussian_score(x, x_fit, sigma):
#     diff = x - x_fit
#     score = jnp.exp(-0.5 * jnp.sum(diff ** 2, axis=1) / (sigma ** 2))
#     return score / jnp.sum(score)


def build_gaussian_score(x_fit, y_fit, sigma):
    x_fit_squared_norm = jnp.sum(x_fit**2, axis=1)

    @jax.jit
    def gaussian_posterior_batch(
        x,
    ):
        """Compute Gaussian-kernel posterior for a batch.

        Parameters
        ----------
        x : jax.Array of shape (n_predict, n_features)
            Prediction features.
        x_fit : jax.Array of shape (n_fit, n_features)
            Training features.
        x_fit_squared_norm : jax.Array of shape (n_fit,)
            Precomputed squared norms of training features.
        y_fit : jax.Array of shape (n_fit, n_bins)
            Training position distributions.
        sigma : float
            Gaussian kernel standard deviation.

        Returns
        -------
        posterior : jax.Array of shape (n_predict, n_bins)
            Predicted spatial posterior.
        """
        x_squared_norm = jnp.sum(
            x**2,
            axis=1,
            keepdims=True,
        )

        squared_distance = (
            x_squared_norm + x_fit_squared_norm[None, :] - 2.0 * x @ x_fit.T
        )

        # Protect against tiny negative values from floating-point error.
        squared_distance = jnp.maximum(
            squared_distance,
            0.0,
        )

        weights = jnp.exp(-0.5 * squared_distance / sigma**2)

        weights /= jnp.sum(
            weights,
            axis=1,
            keepdims=True,
        )

        return weights @ y_fit

    return gaussian_posterior_batch


def build_dot_product_score(x_fit, y_fit, temp=1.0):
    @jax.jit
    def dot_product_posterior_batch(
        x,
    ):
        """Compute dot-product posterior for a batch.

        Parameters
        ----------
        x : jax.Array of shape (n_predict, n_features)
            Prediction features.
        x_fit : jax.Array of shape (n_fit, n_features)
            Training features.
        y_fit : jax.Array of shape (n_fit, n_bins)
            Training position distributions.

        Returns
        -------
        posterior : jax.Array of shape (n_predict, n_bins)
            Predicted spatial posterior.
        """
        weights = jnp.exp(x @ x_fit.T / temp)

        weights /= jnp.sum(
            weights,
            axis=1,
            keepdims=True,
        )

        return weights @ y_fit

    return dot_product_posterior_batch


SCORE_METHODS = {
    "gaussian": build_gaussian_score,
    "dot_product": build_dot_product_score,
}
