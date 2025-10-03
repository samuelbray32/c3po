import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

import jax

from ..model.model import Embedding, C3PO

from flax import serialization


def interval_list_contains_ind(interval_list, timestamps):
    """Find indices of list of timestamps contained in an interval list.

    Parameters
    ----------
    interval_list : array_like
        Each element is (start time, stop time), i.e. an interval in seconds.
    timestamps : array_like
    """
    ind = []
    for interval in interval_list:
        ind += np.ravel(
            np.argwhere(
                np.logical_and(timestamps >= interval[0], timestamps <= interval[1])
            )
        ).tolist()
    return np.asarray(ind)


def interval_list_contains(interval_list, timestamps):
    """Find timestamps that are contained in an interval list.

    Parameters
    ----------
    interval_list : array_like
        Each element is (start time, stop time), i.e. an interval in seconds.
    timestamps : array_like
    """
    ind = []
    for interval in interval_list:
        ind += np.ravel(
            np.argwhere(
                np.logical_and(timestamps >= interval[0], timestamps <= interval[1])
            )
        ).tolist()
    return timestamps[ind]


class C3poAnalysis:
    def __init__(self, model=None, model_args=None, params=None, init_key=None):
        if init_key:
            if not all([val is None for val in [model, model_args, params]]):
                raise ValueError(
                    "If init_key is provided, model, model_args, and params must be null value."
                )
            model, model_args, params = self.initialize_from_table_entry(init_key)
        elif not all([val is not None for val in [model, model_args, params]]):
            raise ValueError(
                "If init_key is not provided, model, model_args, and params must be provided."
            )

        self.model = model
        self.model_args = model_args  # arguments to generate the model
        self.params = params  # parameters of the model (currently assume trained)

        self.z = None  # latent marks of embedded data
        self.c = None  # context of embedded data
        self.t = None  # time of embedded datapoints is unix time
        self.t_interp = None  # time of interpolated context
        self.c_interp = None  # interpolated context
        self.c_pca = None
        self.c_pca_interp = None

        self.latent_dim = model_args["latent_dim"]
        self.context_dim = model_args["context_dim"]

    def initialize_from_table_entry(self, init_key):
        from ..tables.dev_tables import C3POStorage

        entry = (C3POStorage & init_key).fetch1()
        model_args = dict(
            encoder_args=entry["encoder_args"],
            context_args=entry["context_args"],
            rate_args=entry["rate_args"],
            distribution=entry.get("distribution", "poisson"),
            latent_dim=entry["latent_dim"],
            context_dim=entry["context_dim"],
            n_neg_samples=1,
            predicted_sequence_length=1,
            sample_params=None,
        )

        model = C3PO(**model_args)

        x_ = np.zeros((1, 100, entry["input_shape"]))
        delta_t_ = np.zeros(
            (
                1,
                100,
            )
        )
        rand_key = jax.random.PRNGKey(0)
        null_params = model.init(jax.random.PRNGKey(1), x_, delta_t_, rand_key)
        params = serialization.from_bytes(null_params, entry["learned_params"])
        return model, model_args, params

    @property
    def encoder_args(self):
        return self.model_args["encoder_args"]

    @property
    def context_args(self):
        return self.model_args["context_args"]

    @property
    def embedding_params(self):
        return {key: self.params[key]["embedding"] for key in self.params}

    def train_model(self):
        raise NotImplementedError()

    # ----------------------------------------------------------------------------------
    # data embedding tools

    def embed_data(
        self,
        x,
        delta_t,
        first_mark_time: float = 0.0,
        chunk_size: int = 50000,
        chunk_padding: int = 2000,
        delta_t_units: str = "ms",
        store_data: bool = True,
        chunk_data: bool = True,
    ):
        """Embed the full mark series into  Z and C.
        Uses chunking and padding to avoid memory issues.

        Args:
            x (np.ndarray): Sequence of marks. Shape (n_samples, input_dim).
            delta_t (np.ndarray): Sequence of time intervals. Shape (n_samples,).
            first_mark_time (float, optional): Time of the first mark. Defaults to 0.0.
            chunk_size (int, optional): Size of the chunks to embed. Defaults to 50000.
            chunk_padding (int, optional): Padding of the chunks to embed. Defaults to 2000.
            delta_t_units (str, optional): Units of delta_t. Defaults to 'ms'.
        """

        # clear existing stored data to avoid confusion
        if store_data and not all([(val is None) for val in [self.z, self.c, self.t]]):
            Warning("Clearing existing data")
            self.z = None
            self.c = None
            self.t = None
            self.t_interp = None
            self.c_interp = None
            self.c_pca = None
            self.pca_intervals = None
            self.c_pca_interp = None

        z = np.zeros((x.shape[1], self.latent_dim)) * np.nan
        c = np.zeros((x.shape[1], self.context_dim)) * np.nan

        @jax.jit
        def embed_chunk(x, delta_t):
            return Embedding(
                self.encoder_args, self.context_args, self.latent_dim, self.context_dim
            ).apply(self.embedding_params, x, delta_t, jax.random.PRNGKey(0))

        i = chunk_padding

        if chunk_data:
            pbar = tqdm(total=x.shape[1])
            while i < (x.shape[1] - chunk_size):
                z_i, c_i = embed_chunk(
                    x[:, i - chunk_padding : i + chunk_size],
                    delta_t[:, i - chunk_padding : i + chunk_size],
                )
                z[i : i + chunk_size] = z_i[0, chunk_padding:]
                c[i : i + chunk_size] = c_i[0, chunk_padding:]
                i += chunk_size
                pbar.update(chunk_size)
            pbar.close()
        else:
            z, c = embed_chunk(x[None, ...], delta_t[None, ...])
            z = np.array(z[0])
            c = np.array(c[0])
            z[:chunk_padding] = np.nan
            c[:chunk_padding] = np.nan

        # store the markls times in unix time
        t = np.cumsum(delta_t)
        if delta_t_units == "ms":
            t *= 1e-3
        elif delta_t_units not in ["s", "ms"]:
            raise ValueError(f"delta_t_units must be 's' or 'ms' not {delta_t_units}")

        if store_data:
            self.z = np.array(z)
            self.c = np.array(c)
            self.t = t + first_mark_time
        return z, c, t + first_mark_time

    def interpolate_context(self, t: np.ndarray = None):
        """Interpolate the context

        Args:
            t (np.ndarray, optional): . Defaults to None.
        """
        self._check_embedded_data()

        if t is None:
            t = np.arange(self.t[0], self.t[-1], 0.002)
        if (
            self.t_interp is not None
            and self.t_interp.size == t.size
            and np.allclose(self.t_interp, t, atol=0.001)
        ):
            return  # already interpolated to this time
        elif self.t_interp is not None:
            Warning("Re-=making interpolated context to new values")
            self.c_interp = None
            self.c_pca_interp = None

        self.c_interp = np.array(
            [np.interp(t, self.t, self.c[:, i]) for i in range(self.context_dim)]
        ).T
        self.t_interp = t
        self.c_pca_interp = None

    # ----------------------------------------------------------------------------------
    # PCA projection

    def fit_context_pca(self, fit_intervals=None):
        self._check_embedded_data()
        if fit_intervals is None:
            fit_intervals = np.array([[self.t[0], self.t[-1]]])

        fit_ind = interval_list_contains_ind(fit_intervals, self.t)
        data = self.c[fit_ind]
        data = data[~np.isnan(data).any(axis=1)]
        pca = PCA(n_components=self.context_dim)
        pca.fit(data)
        self.pca = pca
        self.pca_intervals = fit_intervals
        self.embed_context_pca()

    def embed_context_pca(self):
        if self.pca is None:
            raise ValueError("pca must first be fit to data")
        self.c_pca_interp = None
        if self.c_pca is None:
            self.c_pca = np.ones_like(self.c) * np.nan
            ind_valid = (~np.isnan(self.c)).any(axis=1)
            self.c_pca[ind_valid] = self.pca.transform(self.c[ind_valid])
        if self.c_interp is not None and self.c_pca_interp is None:
            self.c_pca_interp = np.ones_like(self.c_interp) * np.nan
            ind_valid = (~np.isnan(self.c_interp)).any(axis=1)
            self.c_pca_interp[ind_valid] = self.pca.transform(self.c_interp[ind_valid])

    # ----------------------------------------------------------------------------------
    # Latent factor interpretation tools
    def _select_data(self, pca, interpolated):
        if pca and interpolated:
            t_data = self.t_interp
            c_data = self.c_pca_interp
        elif pca and not interpolated:
            t_data = self.t
            c_data = self.c_pca
        elif not pca and interpolated:
            t_data = self.t_interp
            c_data = self.c_interp
        elif not pca and not interpolated:
            t_data = self.t
            c_data = self.c

        return t_data, c_data

    def bin_context_by_feature(
        self,
        feature: np.ndarray,
        feature_times: np.ndarray,
        bins=None,
        valid_intervals=None,
        pca=False,
        interpolated=False,
    ):
        """
        Bin the context by co-occurring feature values

        Args:
            feature (np.ndarray): Feature values. Shape (n_samples,).
            feature_times (np.ndarray): Times of the feature values. Shape (n_samples,).
            bins (int, optional): Number of bins to use. Defaults to None.
            valid_intervals (np.ndarray, optional): Intervals to consider. Defaults to None.
        """
        self._check_embedded_data()
        t_data, c_data = self._select_data(pca, interpolated)

        if valid_intervals is None:
            valid_intervals = np.array(
                [
                    [
                        max(feature_times[0], t_data[0]),
                        min(feature_times[-1], t_data[-1]),
                    ]
                ]
            )

        if bins is None:
            ind_feature = interval_list_contains_ind(valid_intervals, feature_times)
            bins = np.linspace(
                np.min(feature[ind_feature]), np.max(feature[ind_feature]), 30
            )

        ind_context = interval_list_contains_ind(valid_intervals, t_data)
        context_to_feature_ind = np.digitize(t_data[ind_context], feature_times)
        context_features = feature[context_to_feature_ind]

        context_binned = []
        for i in range(len(bins) - 1):
            ind_bin = np.logical_and(
                context_features > bins[i], context_features < bins[i + 1]
            )
            context_binned.append(c_data[ind_context][ind_bin])
        return context_binned, bins

    def bin_context_by_feature_2d(
        self,
        feature_1: np.ndarray,
        feature_1_times: np.ndarray,
        feature_2: np.ndarray,
        feature_2_times: np.ndarray,
        bins_1=None,
        bins_2=None,
        valid_intervals=None,
        pca=False,
        interpolated=False,
    ):
        """
        Bin the context by co-occurring feature values

        Args:
            feature_1 (np.ndarray): Feature values. Shape (n_samples,).
            feature_1_times (np.ndarray): Times of the feature values. Shape (n_samples,).
            feature_2 (np.ndarray): Feature values. Shape (n_samples,).
            feature_2_times (np.ndarray): Times of the feature values. Shape (n_samples,).
            bins (int, optional): Number of bins to use. Defaults to None.
            valid_intervals (np.ndarray, optional): Intervals to consider. Defaults to None.
        """
        # select and parse data
        self._check_embedded_data()
        t_data, c_data = self._select_data(pca, interpolated)
        if valid_intervals is None:
            valid_intervals = np.array(
                [
                    [
                        np.max([feature_1_times[0], feature_2_times[0], t_data[0]]),
                        np.min([feature_1_times[-1], feature_2_times[-1], t_data[-1]]),
                    ]
                ]
            )

        if bins_1 is None:
            ind_feature_1 = interval_list_contains_ind(valid_intervals, feature_1_times)
            bins_1 = np.linspace(
                np.min(feature_1[ind_feature_1]), np.max(feature_1[ind_feature_1]), 30
            )
        if bins_2 is None:
            ind_feature_2 = interval_list_contains_ind(valid_intervals, feature_2_times)
            bins_2 = np.linspace(
                np.min(feature_2[ind_feature_2]), np.max(feature_2[ind_feature_2]), 30
            )

        ind_context = interval_list_contains_ind(valid_intervals, t_data)

        # map context timepoints to feature values
        context_to_feature_1_ind = np.digitize(t_data[ind_context], feature_1_times)
        context_features_1 = feature_1[context_to_feature_1_ind]
        context_to_feature_2_ind = np.digitize(t_data[ind_context], feature_2_times)
        context_features_2 = feature_2[context_to_feature_2_ind]

        # map feature values to bins
        feature_1_bin = np.digitize(context_features_1, bins_1, right=True) - 1
        feature_2_bin = np.digitize(context_features_2, bins_2, right=True) - 1

        # context_binned = [[[] for _ in range(len(bins_2))] for _ in range(len(bins_1))]
        # for i, (b1, b2) in tqdm(enumerate(zip(feature_1_bin, feature_2_bin))):
        #     context_binned[b1][b2].append(c_data[ind_context][i])
        # return context_binned, bins_1, bins_2

        # Remove out-of-range bins
        import pandas as pd

        # Create a mask to filter values within valid bin ranges
        valid_mask = (
            (feature_1_bin >= 0)
            & (feature_1_bin < len(bins_1))
            & (feature_2_bin >= 0)
            & (feature_2_bin < len(bins_2))
        )

        feature_1_bin = feature_1_bin[valid_mask]
        feature_2_bin = feature_2_bin[valid_mask]

        # values now has shape (n_samples, n_dim)
        values = c_data[ind_context][valid_mask]

        # Create a DataFrame. Convert each row of the 2D array to a list/array.
        df = pd.DataFrame({"b1": feature_1_bin, "b2": feature_2_bin})
        # List conversion so each element is an array of shape (n_dim,)
        df["val"] = list(values)

        # Group by bin indices and stack the arrays for each group.
        grouped = (
            df.groupby(["b1", "b2"])["val"]
            .apply(lambda x: np.stack(x.to_list()))
            .reset_index()
        )

        # Create an empty list-of-lists structure for the binned data.
        max_b1, max_b2 = len(bins_1), len(bins_2)
        context_binned = [[[] for _ in range(max_b2)] for _ in range(max_b1)]

        # Populate the bins using the grouped data.
        for _, row in grouped.iterrows():
            b1, b2 = int(row.b1), int(row.b2)
            context_binned[b1][
                b2
            ] = row.val  # Each is now an array of shape (n_bin, n_dim)

        return context_binned, bins_1, bins_2

    from typing import Optional, Tuple

    def alligned_response(
        self,
        marks: np.ndarray,
        t_plot: tuple[float, float],
        pca=True,
        passed_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """
        Look at the model "response function" around the mark times

        Args:
            marks (np.ndarray): Time points to allign to (in seconds) (ex. stimulus, reward, neuron firing).
            t_plot (tuple(float,float)): Time window to plot around each mark time (ex. (-.1,.5)).
            pca (bool, optional): Use PCA context. Defaults to True.
            passed_data (Optional[Tuple[np.ndarray, np.ndarray]], optional): Passed data. Defaults to None.
        """

        if passed_data is None:
            # interpolate the context if not already done
            try:
                self._check_interpolated_data()
            except ValueError:
                self.interpolate_context()
                self.embed_context_pca()

            c_data = self.c_pca_interp if pca else self.c_interp
            t_data = self.t_interp
        else:
            c_data, t_data = passed_data

        dt = np.median(np.diff(t_data))
        window = int(t_plot[0] / dt), int(t_plot[1] / dt)
        mark_inds = np.digitize(marks, t_data) - 1
        mark_inds = mark_inds[
            np.logical_and(
                mark_inds + window[0] >= 0, mark_inds + window[1] < c_data.shape[0]
            )
        ]
        response = []
        response_ind_array = np.arange(window[0], window[1])
        response_ind_array = response_ind_array[None, :] + mark_inds[:, None]
        response_ind_array = response_ind_array.flatten()
        mark_inds = mark_inds.flatten()
        if response_ind_array.size == 0:
            return np.array([])
        if c_data.ndim == 1:
            response = c_data[response_ind_array].reshape(
                (len(mark_inds), window[1] - window[0])
            )
        else:
            response = c_data[response_ind_array, :].reshape(
                (len(mark_inds), window[1] - window[0], c_data.shape[1])
            )
        return response

    def _check_embedded_data(self):
        if any(val is None for val in [self.z, self.c, self.t]):
            raise ValueError("Data not embedded yet")

    def _check_interpolated_data(self):
        self._check_embedded_data()
        if any(val is None for val in [self.t_interp, self.c_interp]):
            raise ValueError("Data not interpolated yet")

    def save_embedding(self, file_path: str):
        """
        Save the embedded data to a file.

        Args:
            file_path (str): Path to the file where the data will be saved.
        """
        if self.z is None or self.c is None or self.t is None:
            raise ValueError("Data not embedded yet")

        np.savez(file_path, z=self.z, c=self.c, t=self.t)

    def load_embedding(self, file_path: str):
        """
        Load the embedded data from a file.

        Args:
            file_path (str): Path to the file from which the data will be loaded.
        """
        data = np.load(file_path)
        self.z = data["z"]
        self.c = data["c"]
        self.t = data["t"]


def bootstrap_traces(
    data,
    sample_size=None,
    statistic=np.mean,
    n_boot=1e3,
    conf_interval=95,
):
    if sample_size is None:
        sample_size = data.shape[0]
    bootstrap = []
    #     for i in tqdm(range(int(n_boot)),position=0,leave=True):
    for i in range(int(n_boot)):
        bootstrap.append(
            statistic(
                data[np.random.choice(np.arange(data.shape[0]), sample_size), :], axis=0
            )
        )
    bootstrap = np.array(bootstrap)
    return np.mean(bootstrap, axis=0), [
        np.percentile(bootstrap, (100 - conf_interval) / 2, axis=0),
        np.percentile(bootstrap, conf_interval + (100 - conf_interval) / 2, axis=0),
    ]
