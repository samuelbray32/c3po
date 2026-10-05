from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from non_local_detector import Environment
import numpy as np


class DiscretizedRegression:

    def __init__(
        self,
        n_bins=10,
        bin_strategy="uniform",
        max_iter=1000,
        balance_groups=False,
        multidim=False,
        predict_method="weighted",
        environment: Environment = None,
        feature_smooth_std=None,
        indicator_discretizer=False,
        **kwargs,
    ):
        if indicator_discretizer:
            self.discretizer = JointIndicatorDiscretizer(
                n_bins=n_bins, bin_strategy=bin_strategy, environment=environment
            )
            multidim = True
        elif environment is not None:
            self.discretizer = GraphDiscretizer(environment=environment)
        elif multidim:
            self.discretizer = Ordinal2dDiscretizer(
                n_bins=n_bins, bin_strategy=bin_strategy
            )
        else:
            self.discretizer = KBinsDiscretizer(
                n_bins=n_bins, encode="ordinal", strategy=bin_strategy
            )
        self.n_bins = n_bins  # stored for copy
        self.bin_strategy = bin_strategy  # stored for copy
        self.multidim = multidim
        # self.model = LogisticRegression(max_iter=max_iter, **kwargs)
        self.model = JAXLogisticRegression(max_iter=max_iter, **kwargs)
        self.balance_groups = balance_groups
        self.predict_method = predict_method
        self.feature_smooth_std = feature_smooth_std
        self.environment = environment
        self._model_kwargs = kwargs

    def copy(self):
        return DiscretizedRegression(
            n_bins=self.n_bins,
            bin_strategy=self.bin_strategy,
            max_iter=self.model.max_iter,
            balance_groups=self.balance_groups,
            multidim=self.multidim,
            predict_method=self.predict_method,
            feature_smooth_std=self.feature_smooth_std,
            environment=self.environment,
            **self._model_kwargs,
        )

    def fit(self, X, y):
        print(np.nanmin(y), np.nanmax(y))
        # y = np.squeeze(y)
        print(y.shape, X.shape)
        ind = np.logical_and(~np.isnan(y).any(axis=1), np.all(~np.isnan(X), axis=1))
        X = X[ind]
        y = y[ind]
        print(np.min(y), np.max(y))
        y_fit = y.reshape(-1, 1) if not self.multidim else y
        y_binned = self.discretizer.fit_transform(y_fit)

        sample_weights = None
        if self.balance_groups:
            class_counts = np.bincount(y_binned.astype(int).flatten())
            total_counts = len(y_binned)
            class_weights = {
                i: total_counts / (len(class_counts) * count)
                for i, count in enumerate(class_counts)
                if count > 0
            }
            sample_weights = np.empty(len(y_binned), dtype=float)
            for i, count in enumerate(class_counts):
                if count > 0:
                    sample_weights[y_binned == i] = class_weights[i]
            # self.model = LogisticRegression(
            #     max_iter=self.model.max_iter,
            #     class_weight=class_weights,
            #     **self._model_kwargs,
            # )
            # self.model = JAXLogisticRegression(
            #     max_iter=self.model.max_iter,
            #     l2_reg=self.model.l2_reg,
            #     **self._model_kwargs,
            # )
            sample_weights = sample_weights

        if self.feature_smooth_std is None:
            self.model.fit(X, y_binned.flatten(), sample_weight=sample_weights)
            return

        # If feature_smooth_std is provided, compute smoothed labels and expand the dataset
        bin_ids, smoothed_weights = self.discretizer.get_smoothed_labels(
            y_fit, position_std=self.feature_smooth_std
        )
        row_sums = np.nansum(smoothed_weights, axis=1, keepdims=True)
        smoothed_weights = smoothed_weights / row_sums
        ind_valid = ~np.any(np.isnan(smoothed_weights), axis=1)
        smoothed_weights = smoothed_weights[ind_valid]

        self.model.fit(
            X[ind_valid],
            smoothed_weights,
            bin_ids,
            sample_weight=(
                sample_weights[ind_valid] if sample_weights is not None else None
            ),
        )
        # X_expanded, y_expanded, sample_weights = self.discretizer.expand_smoothed_labels(
        #     X, bin_ids, smoothed_weights, min_weight=1e-4
        # )
        # self.model.fit(X_expanded, y_expanded.flatten(), sample_weight=sample_weights)
        return

    def _observed_bin_centers(self):
        if hasattr(self.discretizer, "bin_centers"):
            bin_centers = self.discretizer.bin_centers
        else:
            bin_centers = (
                self.discretizer.bin_edges_[0][:-1] + self.discretizer.bin_edges_[0][1:]
            ) / 2
        return bin_centers[self.model.classes_.astype(int)]

    def predict(self, X, return_posterior=False):
        # y_binned_pred = self.model.predict(X)
        # return self.discretizer.inverse_transform(y_binned_pred[:, None])
        y_binned_probs = self.model.predict_proba(X)
        # if self.multidim:
        #     ordinal_pred = self.model.predict(X)
        #     return self.discretizer.inverse_transform(ordinal_pred[:, None])

        bin_centers = self._observed_bin_centers()
        if self.predict_method == "weighted":
            y_pred = np.dot(y_binned_probs, bin_centers)
        elif self.predict_method == "peak":
            y_pred = bin_centers[np.argmax(y_binned_probs, axis=1)]
        elif self.predict_method == "weighted_angle":
            complex_bin_centers = np.exp(1j * bin_centers)
            y_pred_complex = np.dot(y_binned_probs, complex_bin_centers)
            y_pred = np.angle(y_pred_complex)
        else:
            raise ValueError(
                f"Unknown predict_method: {self.predict_method}. Choose 'weighted', 'peak', or 'weighted_angle'."
            )
        if return_posterior:
            return y_pred[:, None], (y_binned_probs, self._observed_bin_centers())
        return y_pred[:, None]

    def _predict_multidim(self, X):
        y_binned_probs = self.model.predict_proba(X)
        ordinal_pred = np.argmax(y_binned_probs, axis=1)
        return self.discretizer.inverse_transform(ordinal_pred[:, None])


class JointIndicatorDiscretizer:
    def __init__(
        self, n_bins=10, bin_strategy="uniform", environment: Environment = None
    ):
        if environment is not None:
            self.discretizer = GraphDiscretizer(environment=environment)
        else:
            self.discretizer = KBinsDiscretizer(
                n_bins=n_bins, encode="ordinal", strategy=bin_strategy
            )
        self.n_bins = n_bins

    def fit(self, y):
        self.discretizer.fit(y[:, 0][:, None])
        return

    def transform(self, y):
        y_binned = self.discretizer.transform(y[:, 0][:, None])
        y_binned = y_binned + y[:, 1][:, None] * self.n_bins
        return y_binned

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def get_smoothed_labels(self, y, position_std):
        bin_ids, smoothed_weights = self.discretizer.get_smoothed_labels(
            y[:, 0][:, None], position_std
        )
        expanded_smoothed_weights = np.zeros(
            (len(smoothed_weights), len(smoothed_weights[0]) * 2)
        )
        ind_false = np.where(y[:, 1] == 0)[0]
        ind_true = np.where(y[:, 1] == 1)[0]
        expanded_smoothed_weights[ind_false, : len(smoothed_weights[0])] = (
            smoothed_weights[ind_false]
        )
        expanded_smoothed_weights[ind_true, len(smoothed_weights[0]) :] = (
            smoothed_weights[ind_true]
        )
        expanded_bin_ids = np.concatenate([bin_ids, bin_ids + self.n_bins])
        return expanded_bin_ids, expanded_smoothed_weights

    @property
    def bin_centers(self):
        if hasattr(self.discretizer, "bin_centers"):
            bin_centers = self.discretizer.bin_centers
        else:
            bin_centers = (
                self.discretizer.bin_edges_[0][:-1] + self.discretizer.bin_edges_[0][1:]
            ) / 2

        expanded_bin_centers = np.zeros((len(bin_centers) * 2, 2))
        expanded_bin_centers[:, 0] = np.repeat(bin_centers, 2)
        expanded_bin_centers[self.n_bins :, 1] = 1

        return expanded_bin_centers


class Ordinal2dDiscretizer:
    def __init__(self, n_bins=10, bin_strategy="uniform"):
        if isinstance(n_bins, int):
            n_bins = (n_bins, n_bins)
        self.n_bins = n_bins
        self.bin_strategy = bin_strategy
        self.discretizers = [
            KBinsDiscretizer(n_bins=n_bins[i], encode="ordinal", strategy=bin_strategy)
            for i in range(2)
        ]
        self.null_bins = None

    def fit_transform(self, y):
        print(y.shape)
        self.fit(y)
        return self.transform(y)

    def fit(self, y):
        y_binned = np.zeros_like(y)
        for i in range(2):
            y_binned[:, i : i + 1] = self.discretizers[i].fit_transform(
                y[:, i][:, None]
            )
        y_ordinal = y_binned[:, 0] * self.n_bins[1] + y_binned[:, 1]
        ordinal_counts = np.bincount(y_ordinal.astype(int).flatten())
        null_bins = np.where(ordinal_counts == 0)[0]
        if len(null_bins) == 0:
            return

        self.null_bins = null_bins
        self.ordinal_compression_map = {}
        for i in range(len(ordinal_counts)):
            if i in null_bins:
                continue
            self.ordinal_compression_map[i] = len(self.ordinal_compression_map)
        self.ordinal_decompression_map = {
            v: k for k, v in self.ordinal_compression_map.items()
        }
        return

    def transform(self, y):
        y_binned = np.zeros_like(y)
        for i in range(2):
            y_binned[:, i : i + 1] = self.discretizers[i].transform(y[:, i][:, None])

        y_ordinal = y_binned[:, 0] * self.n_bins[1] + y_binned[:, 1]
        if self.null_bins is None:
            return y_ordinal[:, None]
        y_ordinal_compressed = np.array(
            [self.ordinal_compression_map[val] for val in y_ordinal]
        )
        return y_ordinal_compressed[:, None]

    def inverse_transform(self, y_ordinal):
        y_binned = np.zeros((y_ordinal.shape[0], 2))

        if self.null_bins is not None:
            y_ordinal_decompressed = np.array(
                [self.ordinal_decompression_map[val] for val in y_ordinal.flatten()]
            )
            y_ordinal = y_ordinal_decompressed[:, None]
        y_binned[:, 0] = y_ordinal[:, 0] // self.n_bins[1]
        y_binned[:, 1] = y_ordinal[:, 0] % self.n_bins[1]

        y = np.zeros_like(y_binned)
        for i in range(2):
            y[:, i : i + 1] = self.discretizers[i].inverse_transform(
                y_binned[:, i : i + 1]
            )
        return y

    @property
    def bin_centers(self):
        return self.inverse_transform(np.arange(len(self.ordinal_compression_map)))


class GraphDiscretizer:
    def __init__(
        self,
        environment: Environment,
    ):
        self.environment = environment

    def fit_transform(self, y):
        print(y.shape)
        self.fit(y)
        return self.transform(y)

    def fit(self, y):
        return

    def transform(self, y):
        return self.environment.get_bin_ind(y)

    def inverse_transform(self, y_binned):
        raise NotImplementedError(
            "Inverse transform is not implemented for GraphDiscretizer."
        )

    @property
    def bin_centers(self):
        return self.environment.place_bin_centers_.squeeze()

    def interior_bin_ids(self):
        return np.flatnonzero(self.environment.is_track_interior_.ravel())

    def get_interior_bin_id(self, y):
        bin_ids = self.transform(y)
        interior_bin_ids = self.interior_bin_ids()
        interior_bin_id_map = {bin_id: i for i, bin_id in enumerate(interior_bin_ids)}
        interior_bin_ids_for_y = np.array(
            [interior_bin_id_map.get(bin_id, -1) for bin_id in bin_ids]
        )
        return interior_bin_ids_for_y

    def get_smoothed_labels(
        self,
        y: np.ndarray,
        position_std: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Gaussian-smoothed graph-bin labels.

        Parameters
        ----------
        y : ndarray of shape (n_samples, 1)
            Linearized position.
        position_std : float
            Gaussian standard deviation in position units.

        Returns
        -------
        bin_ids : ndarray of shape (n_interior_bins,)
            Environment bin IDs corresponding to the probability columns.
        weights : ndarray of shape (n_samples, n_interior_bins)
            Gaussian-smoothed target weights.
        """
        distances = get_distances_to_interior_bins(self.environment, y)[0]

        weights = np.exp(-0.5 * (distances / position_std) ** 2)
        weights /= weights.sum(axis=1, keepdims=True)

        bin_ids = np.flatnonzero(self.environment.is_track_interior_.ravel())

        return bin_ids, weights

    def expand_smoothed_labels(
        self,
        X: np.ndarray,
        bin_ids: np.ndarray,
        weights: np.ndarray,
        min_weight: float = 1e-4,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Expand observations for soft-label logistic regression.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Decoder features.
        bin_ids : ndarray of shape (n_bins,)
            Spatial bin IDs.
        weights : ndarray of shape (n_samples, n_bins)
            Soft target weights.
        min_weight : float, default=1e-4
            Ignore negligible Gaussian weights.

        Returns
        -------
        X_expanded : ndarray of shape (n_expanded, n_features)
            Repeated decoder features.
        y_expanded : ndarray of shape (n_expanded,)
            Spatial-bin labels.
        sample_weights : ndarray of shape (n_expanded,)
            Gaussian label weights.
        """
        sample_ind, bin_ind = np.nonzero(weights > min_weight)

        X_expanded = X[sample_ind]
        y_expanded = bin_ids[bin_ind]
        sample_weights = weights[sample_ind, bin_ind]

        return X_expanded, y_expanded, sample_weights


def get_distances_to_interior_bins(
    environment: Environment,
    position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute graph distances from positions to all interior bins.

    Parameters
    ----------
    environment : Environment
        Fitted graph-based non_local_detector Environment.
    position : ndarray of shape (n_samples,) or (n_samples, 1)
        Linearized position.

    Returns
    -------
    distances : ndarray of shape (n_samples, n_interior_bins)
        Shortest-path graph distance from each position bin to each
        interior spatial bin.
    interior_bin_ids : ndarray of shape (n_interior_bins,)
        Environment bin IDs corresponding to columns of ``distances``.
    """
    position = np.asarray(position, dtype=float).reshape(-1, 1)

    position_bin_ids = environment.get_bin_ind(position)

    interior_bin_ids = np.flatnonzero(environment.is_track_interior_.ravel())
    # interior_bin_ids = np.flatnonzero(
    #     np.ones_like(environment.is_track_interior_.ravel(), dtype=bool)
    # )

    bin_to_node = np.asarray(
        environment.place_bin_centers_nodes_df_.node_id,
        dtype=int,
    )

    source_nodes = bin_to_node[position_bin_ids]
    target_nodes = bin_to_node[interior_bin_ids]

    distances = np.full(
        (len(position), len(interior_bin_ids)),
        np.inf,
        dtype=float,
    )

    for row, source_node in enumerate(source_nodes):
        if source_node == -1:
            continue

        source_distances = environment.distance_between_nodes_[source_node]

        distances[row] = [
            source_distances.get(target_node, np.inf) for target_node in target_nodes
        ]

    return distances, interior_bin_ids


def graph_distance_between_points(
    environment: Environment,
    position_a: np.ndarray,
    position_b: np.ndarray,
) -> np.ndarray:
    """Compute graph distances between two sets of positions.

    Parameters
    ----------
    environment : Environment
        Fitted graph-based non_local_detector Environment.
    position_a : ndarray of shape (n_samples,) or (n_samples, 1)
        Linearized position A.
    position_b : ndarray of shape (n_samples,) or (n_samples, 1)
        Linearized position B.

    Returns
    -------
    distances : ndarray of shape (n_samples,)
        Shortest-path graph distance from each position in A to the
        corresponding position in B at the same sample index.
    """
    position_a = np.asarray(position_a, dtype=float).reshape(-1, 1)
    position_b = np.asarray(position_b, dtype=float).reshape(-1, 1)

    if len(position_a) != len(position_b):
        raise ValueError(
            "position_a and position_b must contain the same number of samples."
        )

    bin_ids_a = environment.get_bin_ind(position_a)
    bin_ids_b = environment.get_bin_ind(position_b)

    bin_to_node = np.asarray(
        environment.place_bin_centers_nodes_df_.node_id,
        dtype=int,
    )

    source_nodes = bin_to_node[bin_ids_a]
    target_nodes = bin_to_node[bin_ids_b]

    distances = np.full(len(position_a), np.inf, dtype=float)

    for row, (source_node, target_node) in enumerate(zip(source_nodes, target_nodes)):
        if source_node == -1 or target_node == -1:
            continue

        source_distances = environment.distance_between_nodes_[source_node]
        distances[row] = source_distances.get(target_node, np.inf)

    return distances


# KNN ----------------------------------------------------------------------------------
class CircularKNNRegressor:
    def __init__(self, **kwargs):
        self.model = KNeighborsRegressor(**kwargs)

    def fit(self, X, y):
        y_circular = np.exp(1j * y.flatten())
        y_circular = np.column_stack((y_circular.real, y_circular.imag))
        self.model.fit(X, y_circular)

    def predict(self, X):
        y_circular_pred = self.model.predict(X)
        y_circular_pred = y_circular_pred[:, 0] + 1j * y_circular_pred[:, 1]
        return np.angle(y_circular_pred)[:, None]


class PosteriorKNN:
    def __init__(
        self,
        n_bins=10,
        bin_strategy="uniform",
        multidim=False,
        predict_method="weighted",
        environment: Environment = None,
        feature_smooth_std=None,
        **kwargs,
    ):
        if environment is not None:
            self.discretizer = GraphDiscretizer(environment=environment)
        elif multidim:
            self.discretizer = Ordinal2dDiscretizer(
                n_bins=n_bins, bin_strategy=bin_strategy
            )
        else:
            self.discretizer = KBinsDiscretizer(
                n_bins=n_bins, encode="onehot", strategy=bin_strategy
            )
        self.n_bins = n_bins  # stored for copy
        self.bin_strategy = bin_strategy  # stored for copy
        self.multidim = multidim
        self.model = KNeighborsRegressor(**kwargs)
        self.predict_method = predict_method
        self.feature_smooth_std = feature_smooth_std
        self.environment = environment
        self._model_kwargs = kwargs

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

        self.model.fit(X, y_binned)

    def _observed_bin_centers(self):
        if hasattr(self.discretizer, "bin_centers"):
            bin_centers = self.discretizer.bin_centers
        else:
            bin_centers = (
                self.discretizer.bin_edges_[0][:-1] + self.discretizer.bin_edges_[0][1:]
            ) / 2
        return bin_centers

    def predict(self, X, return_posterior=False):
        y_binned_probs = self.model.predict(X)
        if self.multidim:
            ordinal_pred = np.argmax(y_binned_probs, axis=1)
            return self.discretizer.inverse_transform(ordinal_pred[:, None])

        bin_centers = self._observed_bin_centers()
        if self.predict_method == "weighted":
            y_pred = np.dot(y_binned_probs, bin_centers)
        elif self.predict_method == "peak":
            y_pred = bin_centers[np.argmax(y_binned_probs, axis=1)]
        elif self.predict_method == "weighted_angle":
            complex_bin_centers = np.exp(1j * bin_centers)
            y_pred_complex = np.dot(y_binned_probs, complex_bin_centers)
            y_pred = np.angle(y_pred_complex)
        else:
            raise ValueError(
                f"Unknown predict_method: {self.predict_method}. Choose 'weighted', 'peak', or 'weighted_angle'."
            )
        if return_posterior:
            return y_pred[:, None], (y_binned_probs, self._observed_bin_centers())
        return y_pred[:, None]


# JAX logistic regression -------------------------------------------------------------
import jax
import jax.numpy as jnp
import numpy as np
import optax
from numpy.typing import NDArray

import jax
import jax.numpy as jnp
import numpy as np
import optax
from numpy.typing import NDArray


class JAXLogisticRegression:
    """Multinomial logistic regression implemented with JAX and Optax."""

    def __init__(
        self,
        learning_rate: float = 1e-2,
        max_iter: int = 1000,
        l2_reg: float = 1e-4,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l2_reg = l2_reg

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating] | NDArray[np.integer],
        classes: NDArray[np.integer] | None = None,
        sample_weight: NDArray[np.floating] | None = None,
    ) -> "JAXLogisticRegression":
        """Fit multinomial logistic regression.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.
        y : ndarray
            Either hard labels with shape (n_samples,) or soft labels with
            shape (n_samples, n_classes).
        classes : ndarray of shape (n_classes,), optional
            Original class IDs corresponding to the columns of soft labels.
            Required when ``y`` contains soft labels.
        sample_weight : ndarray of shape (n_samples,), optional
            Weight assigned to each training observation.
        """
        try:
            print("Attempting full fit...")
            return self.fit_full(X, y, classes, sample_weight)
        except RuntimeError as e:
            if "Resource exhausted" in str(e):
                print(
                    "Resource exhausted during full fit. "
                    "Falling back to chunked fit."
                )
            else:
                raise e
        return self.fit_chunked(X, y, classes, sample_weight)

    def fit_full(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating] | NDArray[np.integer],
        classes: NDArray[np.integer] | None = None,
        sample_weight: NDArray[np.floating] | None = None,
    ) -> "JAXLogisticRegression":
        """Fit multinomial logistic regression.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.
        y : ndarray
            Either hard labels with shape (n_samples,) or soft labels with
            shape (n_samples, n_classes).
        classes : ndarray of shape (n_classes,), optional
            Original class IDs corresponding to the columns of soft labels.
            Required when ``y`` contains soft labels.
        sample_weight : ndarray of shape (n_samples,), optional
            Weight assigned to each training observation.

        Returns
        -------
        self : JAXLogisticRegression
            Fitted model.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        if y.ndim == 1:
            self.classes_ = np.unique(y)

            class_to_index = {
                class_id: index for index, class_id in enumerate(self.classes_)
            }

            y_indices = np.asarray(
                [class_to_index[class_id] for class_id in y],
                dtype=np.int32,
            )

            y_targets = jax.nn.one_hot(
                jnp.asarray(y_indices),
                len(self.classes_),
            )

        elif y.ndim == 2:
            if classes is None:
                raise ValueError("classes must be provided when using soft labels.")

            classes = np.asarray(classes)

            if y.shape[1] != len(classes):
                raise ValueError(
                    "Number of soft-label columns must equal " "the number of classes."
                )

            if np.any(y < 0):
                raise ValueError("Soft labels must be non-negative.")

            row_sums = y.sum(axis=1)

            if not np.allclose(row_sums, 1.0):
                raise ValueError("Each row of soft labels must sum to 1.")

            self.classes_ = classes
            y_targets = jnp.asarray(y, dtype=jnp.float32)

        else:
            raise ValueError(
                "y must have shape (n_samples,) for hard labels or "
                "(n_samples, n_classes) for soft labels."
            )

        X_jax = jnp.asarray(X, dtype=jnp.float32)

        if sample_weight is None:
            sample_weight_jax = jnp.ones(X.shape[0], dtype=jnp.float32)
        else:
            sample_weight_jax = jnp.asarray(
                sample_weight,
                dtype=jnp.float32,
            )

        n_features = X.shape[1]
        n_classes = len(self.classes_)

        params = {
            "W": jnp.zeros(
                (n_features, n_classes),
                dtype=jnp.float32,
            ),
            "b": jnp.zeros(
                n_classes,
                dtype=jnp.float32,
            ),
        }

        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(params)

        def loss_fn(params):
            l2_loss = 0.5 * self.l2_reg * jnp.sum(params["W"] ** 2)

            logits = X_jax @ params["W"] + params["b"]

            losses = optax.softmax_cross_entropy(
                logits=logits,
                labels=y_targets,
            )

            data_loss = jnp.sum(losses * sample_weight_jax) / jnp.sum(sample_weight_jax)

            return data_loss + l2_loss

        value_and_grad = jax.value_and_grad(loss_fn)

        def step(_, carry):
            params, opt_state = carry

            _, grads = value_and_grad(params)

            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                params,
            )

            params = optax.apply_updates(params, updates)

            return params, opt_state

        @jax.jit
        def optimize(params, opt_state):
            return jax.lax.fori_loop(
                0,
                self.max_iter,
                step,
                (params, opt_state),
            )

        params, _ = optimize(params, opt_state)

        self._params = params

        # sklearn-compatible orientation:
        # (n_classes, n_features)
        self.coef_ = np.asarray(params["W"].T)
        self.intercept_ = np.asarray(params["b"])

        return self

    def predict_proba(
        self,
        X: NDArray[np.floating],
    ) -> NDArray[np.float32]:
        """Predict class probabilities.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes)
            Predicted probabilities.
        """
        X_jax = jnp.asarray(X, dtype=jnp.float32)

        logits = X_jax @ self._params["W"] + self._params["b"]

        return np.asarray(jax.nn.softmax(logits, axis=-1))

    def fit_chunked(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating] | NDArray[np.integer],
        classes: NDArray[np.integer] | None = None,
        sample_weight: NDArray[np.floating] | None = None,
        batch_size: int = 8192,
    ) -> "JAXLogisticRegression":
        """Fit multinomial logistic regression in chunks.

        Unlike ``fit``, the complete training arrays are never transferred to
        JAX at once. ``X``, ``y``, and ``sample_weight`` only need to support
        ``shape`` and slicing, so disk-backed arrays such as ``np.memmap`` can
        be used.

        ``max_iter`` is interpreted as the number of passes over the dataset.
        Parameters are updated once per chunk.
        """
        if y.ndim == 1:
            # Find the classes without materializing all of y.
            unique_classes = []

            for start in range(0, y.shape[0], batch_size):
                stop = min(start + batch_size, y.shape[0])
                unique_classes.append(np.unique(np.asarray(y[start:stop])))

            self.classes_ = np.unique(np.concatenate(unique_classes))

            class_to_index = {
                class_id: index for index, class_id in enumerate(self.classes_)
            }

            soft_labels = False

        elif y.ndim == 2:
            if classes is None:
                raise ValueError("classes must be provided when using soft labels.")

            classes = np.asarray(classes)

            if y.shape[1] != len(classes):
                raise ValueError(
                    "Number of soft-label columns must equal " "the number of classes."
                )

            # Validate soft labels chunk-by-chunk.
            for start in range(0, y.shape[0], batch_size):
                stop = min(start + batch_size, y.shape[0])

                y_batch = np.asarray(
                    y[start:stop],
                    dtype=np.float32,
                )

                if np.any(y_batch < 0):
                    raise ValueError("Soft labels must be non-negative.")

                row_sums = y_batch.sum(axis=1)

                if not np.allclose(row_sums, 1.0):
                    raise ValueError("Each row of soft labels must sum to 1.")

            self.classes_ = classes
            soft_labels = True

        else:
            raise ValueError(
                "y must have shape (n_samples,) for hard labels or "
                "(n_samples, n_classes) for soft labels."
            )

        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_classes = len(self.classes_)

        params = {
            "W": jnp.zeros(
                (n_features, n_classes),
                dtype=jnp.float32,
            ),
            "b": jnp.zeros(
                n_classes,
                dtype=jnp.float32,
            ),
        }

        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(params)

        def loss_fn(
            params,
            X_batch,
            y_targets,
            sample_weight_batch,
        ):
            l2_loss = 0.5 * self.l2_reg * jnp.sum(params["W"] ** 2)

            logits = X_batch @ params["W"] + params["b"]

            losses = optax.softmax_cross_entropy(
                logits=logits,
                labels=y_targets,
            )

            data_loss = jnp.sum(losses * sample_weight_batch) / jnp.sum(
                sample_weight_batch
            )

            return data_loss + l2_loss

        value_and_grad = jax.value_and_grad(loss_fn)

        @jax.jit
        def step(
            params,
            opt_state,
            X_batch,
            y_targets,
            sample_weight_batch,
        ):
            _, grads = value_and_grad(
                params,
                X_batch,
                y_targets,
                sample_weight_batch,
            )

            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                params,
            )

            params = optax.apply_updates(
                params,
                updates,
            )

            return params, opt_state

        for _ in range(self.max_iter):
            for start in range(0, n_samples, batch_size):
                stop = min(start + batch_size, n_samples)

                X_batch = jnp.asarray(
                    X[start:stop],
                    dtype=jnp.float32,
                )

                if soft_labels:
                    y_targets = jnp.asarray(
                        y[start:stop],
                        dtype=jnp.float32,
                    )
                else:
                    y_batch = np.asarray(y[start:stop])

                    y_indices = np.asarray(
                        [class_to_index[class_id] for class_id in y_batch],
                        dtype=np.int32,
                    )

                    y_targets = jax.nn.one_hot(
                        jnp.asarray(y_indices),
                        n_classes,
                    )

                if sample_weight is None:
                    sample_weight_batch = jnp.ones(
                        stop - start,
                        dtype=jnp.float32,
                    )
                else:
                    sample_weight_batch = jnp.asarray(
                        sample_weight[start:stop],
                        dtype=jnp.float32,
                    )

                # Avoid dividing by zero for an all-zero-weight chunk.
                if sample_weight is not None:
                    if not np.any(np.asarray(sample_weight[start:stop])):
                        continue

                params, opt_state = step(
                    params,
                    opt_state,
                    X_batch,
                    y_targets,
                    sample_weight_batch,
                )

        self._params = params

        # sklearn-compatible orientation:
        # (n_classes, n_features)
        self.coef_ = np.asarray(params["W"].T)
        self.intercept_ = np.asarray(params["b"])

        return self

    def predict(
        self,
        X: NDArray[np.floating],
    ) -> NDArray[np.integer]:
        """Predict class labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted class IDs.
        """
        probabilities = self.predict_proba(X)
        class_ind = np.argmax(probabilities, axis=1)

        return self.classes_[class_ind]
