import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

import jax
from functools import lru_cache
from multiprocessing import Pool
from scipy.signal import correlate
from typing import Tuple, List

from flax import serialization

from ..model.model import Embedding, C3PO


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

        self.decoder_model = None
        self.decode_pca = None

    def copy(self):
        new_analysis = C3poAnalysis(
            model=self.model,
            model_args=self.model_args,
            params=self.params,
        )

        new_analysis.z = None if self.z is None else self.z.copy()
        new_analysis.c = None if self.c is None else self.c.copy()
        new_analysis.t = None if self.t is None else self.t.copy()
        new_analysis.t_interp = None if self.t_interp is None else self.t_interp.copy()
        new_analysis.c_interp = None if self.c_interp is None else self.c_interp.copy()
        new_analysis.c_pca = None if self.c_pca is None else self.c_pca.copy()
        new_analysis.c_pca_interp = (
            None if self.c_pca_interp is None else self.c_pca_interp.copy()
        )

        new_analysis.pca = getattr(self, "pca", None)

        new_analysis.decoder_model = (
            None if self.decoder_model is None else self.decoder_model
        )
        new_analysis.decode_pca = self.decode_pca

        return new_analysis

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
        if entry["encoder_args"].get("input_format", None) == "indices":
            x_ = x_.astype(np.int16)

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
        if True:  # self.c_pca is None:
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
        alt_data=None,
    ):
        """
        Bin the context by co-occurring feature values

        Args:
            feature (np.ndarray): Feature values. Shape (n_samples,).
            feature_times (np.ndarray): Times of the feature values. Shape (n_samples,).
            bins (int, optional): Number of bins to use. Defaults to None.
            valid_intervals (np.ndarray, optional): Intervals to consider. Defaults to None.
            alt_data (np.ndarray, optional): Alternative data to bin instead of context. Defaults to None.
        """
        if alt_data is None:
            self._check_embedded_data()
            t_data, c_data = self._select_data(pca, interpolated)
        else:
            t_data, c_data = alt_data

        if valid_intervals is None:
            valid_intervals = np.array(
                [
                    [
                        max(feature_times[0], t_data[0]),
                        min(feature_times[-1], t_data[-1]),
                    ]
                ]
            )

        if bins is None or isinstance(bins, int):
            n_bins = 30 if bins is None else bins
            ind_feature = interval_list_contains_ind(valid_intervals, feature_times)
            bins = np.linspace(
                np.min(feature[ind_feature]), np.max(feature[ind_feature]), n_bins
            )

        ind_context = interval_list_contains_ind(valid_intervals, t_data)
        context_to_feature_ind = np.digitize(t_data[ind_context], feature_times) - 1
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
        return_counts=False,
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

        if bins_1 is None or isinstance(bins_1, int):
            n_bins_1 = 30 if bins_1 is None else bins_1
            ind_feature_1 = interval_list_contains_ind(valid_intervals, feature_1_times)
            bins_1 = np.linspace(
                np.min(feature_1[ind_feature_1]),
                np.max(feature_1[ind_feature_1]),
                n_bins_1,
            )
        if bins_2 is None or isinstance(bins_2, int):
            n_bins_2 = 30 if bins_2 is None else bins_2
            ind_feature_2 = interval_list_contains_ind(valid_intervals, feature_2_times)
            bins_2 = np.linspace(
                np.min(feature_2[ind_feature_2]),
                np.max(feature_2[ind_feature_2]),
                n_bins_2,
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

        if not return_counts:
            return context_binned, bins_1, bins_2

        else:
            counts = np.zeros((len(bins_1), len(bins_2)), dtype=int)
            for b1 in range(len(bins_1)):
                for b2 in range(len(bins_2)):
                    counts[b1, b2] = len(context_binned[b1][b2])
            return context_binned, bins_1, bins_2, counts

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
        max_vals = np.max(response, axis=1).max(axis=1)
        return response[~np.isnan(max_vals)]

    # ----------------------------------------------------------------------------------
    # Frequency analysis tools
    def power_spectrum(
        self,
        intervals: np.ndarray = None,
        window_size=1000,
        pca=True,
        nfft=10000,
        sourced_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """Compute the power spectrum of the context using welch's method

        Parameters:
        ----------
            intervals (np.ndarray, optional): Time intervals to compute the power spectrum for. Defaults to None.
            fs (float, optional): Sampling frequency. Defaults to 500.0.
            pca (bool, optional): Whether to use PCA for dimensionality reduction. Defaults to True.

        """
        t, c = (
            self._select_data(pca, interpolated=True)
            if sourced_data is None
            else sourced_data
        )
        ind_valid = (~np.isnan(c)).any(axis=1)
        t = t[ind_valid]
        c = c[ind_valid, :]

        fs = np.mean(np.diff(t)) ** -1

        if intervals is None:
            intervals = np.array([[t[0], t[-1]]])

        from scipy.signal import welch
        from scipy.signal import windows

        window_filt = windows.hamming(window_size, sym=True)
        noverlap = window_size // 2

        mid = []
        lo = []
        hi = []
        for pc in range(c.shape[1]):
            spectrums = []
            weights = []
            for i, interval in enumerate(intervals):
                ind = np.where(np.logical_and(t >= interval[0], t <= interval[1]))[0]
                if len(ind) < window_size:
                    continue
                c_i = c[ind, pc]

                frequencies, P_i = welch(
                    c_i,
                    fs=fs,
                    window=window_filt,
                    noverlap=noverlap,
                    scaling="density",
                    nfft=nfft,
                )
                weight_i = np.floor(len(c) - window_size / (window_size - noverlap)) + 1
                spectrums.append(P_i)
                weights.append(weight_i)
            if len(spectrums) == 1:
                mid.append(spectrums[0])
                lo.append(spectrums[0])
                hi.append(spectrums[0])
                continue
            results = weighted_quantile(
                np.array(spectrums),
                quantiles=[0.5, 0.25, 0.75],
                sample_weight=np.array(weights),
            )

            mid.append(results[0])
            lo.append(results[1])
            hi.append(results[2])

        return frequencies, np.array(mid), np.array(lo), np.array(hi)

    def cross_correlation(
        self, pca=True, max_lag_seconds=1.0, intervals=None, dim_limit=None, processes=1
    ):
        t, c = self._select_data(pca, interpolated=True)
        ind_valid = (~np.isnan(c)).any(axis=1)
        t = t[ind_valid]
        c = c[ind_valid, :]

        fs = np.mean(np.diff(t)) ** -1
        max_lag = int(max_lag_seconds * fs)
        from scipy.signal import correlate

        n_dim = c.shape[1]

        if intervals is None:
            intervals = np.array([[t[0], t[-1]]])
        if dim_limit is not None:
            n_dim = min(n_dim, dim_limit)

        cross_corrs = np.zeros((n_dim, n_dim, 2 * max_lag + 1))

        if processes == 1:
            for i in range(n_dim):
                for j in tqdm(range(n_dim)):
                    cross_corrs[i, j, :] = _single_cross_correlation(
                        c, t, i, j, intervals, max_lag
                    )[0]
                # corr_ij = []
                # weights = []
                # for interval in intervals:
                #     ind = np.where(np.logical_and(t >= interval[0], t <= interval[1]))[
                #         0
                #     ]
                #     if len(ind) < max_lag * 2:
                #         continue
                #     c_i = c[ind, i]
                #     c_j = c[ind, j]
                #     corr_full = correlate(
                #         c_i - np.mean(c_i), c_j - np.mean(c_j), mode="full"
                #     )
                #     mid = len(corr_full) // 2
                #     corr_ij.append(corr_full[mid - max_lag : mid + max_lag + 1])
                #     weights.append(len(c_i) - max_lag)
                # cross_corrs[i, j, :] = np.average(corr_ij, axis=0, weights=weights)

        else:
            # with Pool(processes=processes) as pool:
            #     args = []
            #     for i in range(n_dim):
            #         for j in range(n_dim):
            #             args.append((c, t, i, j, intervals, max_lag))
            #     results = []
            #     for r in tqdm(pool.imap_unordered(_single_cross_correlation, args)):
            #         results.append(r)
            #     for idx, result in enumerate(results):
            #         i = result[1]
            #         j = result[2]
            #         cross_corrs[i, j, :] = result[0]
            args = []
            for i in range(n_dim):
                for j in range(n_dim):
                    args.append((i, j, intervals, max_lag))
            n_tasks = len(args)
            with Pool(
                processes=processes,
                initializer=_init_cc_worker,
                initargs=(c, t),
            ) as pool:
                # Tune chunksize so each worker gets a batch of tasks
                chunksize = max(1, n_tasks // (processes * 4))
                results = []
                for r in tqdm(
                    pool.imap_unordered(
                        _single_cross_correlation_worker, args, chunksize
                    ),
                    total=n_tasks,
                ):
                    results.append(r)
            for idx, result in enumerate(results):
                i = result[1]
                j = result[2]
                cross_corrs[i, j, :] = result[0]

        lags = np.arange(-max_lag, max_lag + 1) / fs
        return lags, cross_corrs

    # ----------------------------------------------------------------------------------
    # decoder tools: Testing decodability of latent factors to known variables
    import sklearn

    def initialize_decoder(
        self, model_type="knn", feature_prediction_delay=0, **kwargs
    ):
        self.feature_prediction_delay = feature_prediction_delay
        self.decoder_model = None
        if isinstance(model_type, str):
            if model_type == "knn":
                from sklearn.neighbors import KNeighborsRegressor

                self.decoder_model = KNeighborsRegressor(**kwargs)
            elif model_type == "linear":
                from sklearn.linear_model import LinearRegression

                self.decoder_model = LinearRegression(**kwargs)

            elif model_type == "discretized_regression":
                from .decoder_models import DiscretizedRegression

                self.decoder_model = DiscretizedRegression(**kwargs)

            else:
                raise ValueError(f"Unknown model type {model_type}")

    def fit_decoder(
        self,
        feature_values,
        feature_times,
        intervals=None,
        pca=True,
        decode_dim: slice = slice(None),
        interpolate=False,
        smooth_context: int = None,
        balance_features=False,
        balance_features_bins=10,
        balance_features_min_count: int = 50,
    ):
        if self.decoder_model is None:
            raise ValueError("Decoder model not initialized")

        # get context data
        self._check_embedded_data()
        if smooth_context:
            t_data, c_data = self._smooth_context(
                pca=pca, interpolated=interpolate, sigma=smooth_context
            )
        else:
            t_data, c_data = self._select_data(pca, interpolated=interpolate)
        c_data = c_data[:, decode_dim]

        # Get feature data
        feature_times = feature_times.copy() - self.feature_prediction_delay

        if intervals is None:
            intervals = np.array([[t_data[0], t_data[-1]]])

        ind_valid = interval_list_contains_ind(intervals, feature_times)
        feature_times = feature_times[ind_valid]
        feature_values = feature_values[ind_valid]

        ind = np.digitize(feature_times, t_data) - 1
        ind_contained = np.logical_and(ind >= 0, ind < t_data.shape[0])
        ind = ind[ind_contained]
        feature_times = feature_times[ind_contained]
        feature_values = feature_values[ind_contained]
        c_data = c_data[ind]

        ind = np.where(~np.isnan(c_data).any(axis=1))[0]
        c_data = c_data[ind]
        feature_times = feature_times[ind]
        feature_values = feature_values[ind]

        ind = np.where(~np.isnan(feature_values).any(axis=1))[0]
        c_data = c_data[ind]
        feature_times = feature_times[ind]
        feature_values = feature_values[ind]

        if balance_features and (
            feature_values.ndim == 1 or feature_values.shape[1] == 1
        ):
            bins = np.linspace(
                np.min(feature_values),
                np.max(feature_values),
                balance_features_bins + 1,
            )
            feature_bin_index = np.digitize(feature_values, bins) - 1
            min_count = min(
                [
                    np.sum(feature_bin_index == i)
                    for i in range(len(bins) - 1)
                    if np.sum(feature_bin_index == i) > 0
                ]
            )
            min_count = max(min_count, balance_features_min_count)
            c_data_balanced = []
            feature_values_balanced = []
            for i in range(len(bins) - 1):
                bin_indices = np.where(feature_bin_index == i)[0]
                if len(bin_indices) == 0:
                    continue
                selected_indices = np.random.choice(
                    bin_indices, size=min(min_count, len(bin_indices)), replace=False
                )
                c_data_balanced.append(c_data[selected_indices])
                feature_values_balanced.append(feature_values[selected_indices])
            c_data = np.vstack(c_data_balanced)
            feature_values = np.vstack(feature_values_balanced)

        elif (
            balance_features and feature_values.ndim > 1 and feature_values.shape[1] > 1
        ):
            bins_1 = np.linspace(
                np.min(feature_values[:, 0]),
                np.max(feature_values[:, 0]),
                balance_features_bins + 1,
            )
            bins_2 = np.linspace(
                np.min(feature_values[:, 1]),
                np.max(feature_values[:, 1]),
                balance_features_bins + 1,
            )
            feature_bin_index_1 = np.digitize(feature_values[:, 0], bins_1) - 1
            feature_bin_index_2 = np.digitize(feature_values[:, 1], bins_2) - 1
            feature_bin_index = feature_bin_index_1 * len(bins_2) + feature_bin_index_2
            min_count = min(
                [
                    np.sum(feature_bin_index == i)
                    for i in range(np.max(feature_bin_index) + 1)
                    if np.sum(feature_bin_index == i) > 0
                ]
            )
            min_count = max(min_count, balance_features_min_count)
            c_data_balanced = []
            feature_values_balanced = []
            for i in range(np.max(feature_bin_index) + 1):
                bin_indices = np.where(feature_bin_index == i)[0]
                if len(bin_indices) == 0:
                    continue
                selected_indices = np.random.choice(
                    bin_indices, size=min(min_count, len(bin_indices)), replace=False
                )
                c_data_balanced.append(c_data[selected_indices])
                feature_values_balanced.append(feature_values[selected_indices])
            c_data = np.vstack(c_data_balanced)
            feature_values = np.vstack(feature_values_balanced)

        self.decoder_model.fit(c_data, feature_values)
        self.decode_pca = pca
        self.decode_dim = decode_dim

    def predict_decoder(self, interval, interpolate=False, smooth_context: int = None):
        self._check_embedded_data()
        if self.decoder_model is None:
            raise ValueError("Decoder model not initialized")

        if smooth_context:
            t_data, c_data = self._smooth_context(
                pca=self.decode_pca, interpolated=interpolate, sigma=smooth_context
            )
        else:
            t_data, c_data = self._select_data(
                self.decode_pca, interpolated=interpolate
            )

        c_data = c_data[:, self.decode_dim]

        ind = np.where(np.logical_and(t_data >= interval[0], t_data <= interval[1]))[0]
        if len(ind) == 0:
            return np.array([]), np.array([])
        ind = ind[~np.isnan(c_data[ind]).any(axis=1)]
        if len(ind) == 0:
            return np.array([]), np.array([])
        return t_data[ind] + self.feature_prediction_delay, self.decoder_model.predict(
            c_data[ind]
        )

    # ----------------------------------------------------------------------------------
    # Utilities
    def _check_embedded_data(self):
        if any(val is None for val in [self.z, self.c, self.t]):
            raise ValueError("Data not embedded yet")

    def _check_interpolated_data(self):
        self._check_embedded_data()
        if any(val is None for val in [self.t_interp, self.c_interp]):
            raise ValueError("Data not interpolated yet")

    @lru_cache(maxsize=1)
    def _smooth_context(self, pca: bool, interpolated: bool, sigma: int):
        from scipy.ndimage import gaussian_filter1d

        self._check_embedded_data()
        t_data, c_data = self._select_data(pca, interpolated=interpolated)
        c_smooth = gaussian_filter1d(c_data, sigma, axis=0, mode="nearest")
        return t_data, c_smooth

    # ----------------------------------------------------------------------------------
    # Save/load embedded data
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


def weighted_quantile(
    values,
    quantiles,
    sample_weight=None,
):
    """Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    if values.ndim == 1:
        return _weighted_quantile_single(
            values,
            quantiles,
            sample_weight=sample_weight,
        )
    else:
        return np.array(
            [
                _weighted_quantile_single(
                    values[:, i],
                    quantiles,
                    sample_weight=sample_weight,
                )
                for i in range(values.shape[1])
            ]
        ).T


def _weighted_quantile_single(
    values,
    quantiles,
    sample_weight=None,
):
    """Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(
        quantiles <= 1
    ), "quantiles should be in [0, 1]"

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight

    weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def _single_cross_correlation(c, t, i, j, intervals, max_lag):
    """Returns the cross-correlation between dimensions i and j of c over the specified intervals.

    Parameters
    ----------
    c : np.ndarray
        Context array of shape (n_samples, n_dimensions).
    t : np.ndarray
        Time array of shape (n_samples,).
    i : int
        First dimension index.
    j : int
        Second dimension index.
    intervals : (n_intervals, 2)
        Intervals array.
    max_lag : int
        Maximum lag (indices).

    Returns
    -------
    result : tuple
        (cross_corr, i, j)
        cross_corr : (2 * max_lag + 1,)
            Cross-correlation values.
        i : int
            First dimension index.
        j : int
            Second dimension index.
    """
    corr_ij = []
    weights = []
    for interval in intervals:
        ind = np.where(np.logical_and(t >= interval[0], t <= interval[1]))[0]
        if len(ind) < max_lag * 2:
            continue
        c_i = c[ind, i]
        c_j = c[ind, j]
        corr_full = correlate(c_i - np.mean(c_i), c_j - np.mean(c_j), mode="full")
        mid = len(corr_full) // 2
        corr_ij.append(corr_full[mid - max_lag : mid + max_lag + 1])
        weights.append(len(c_i) - max_lag)
    return np.average(corr_ij, axis=0, weights=weights), i, j


_c_global: np.ndarray | None = None
_t_global: np.ndarray | None = None


def _init_cc_worker(c: np.ndarray, t: np.ndarray) -> None:
    """Initializer that runs once in each worker process."""
    global _c_global, _t_global
    _c_global = c
    _t_global = t


def _single_cross_correlation_worker(
    args: Tuple[int, int, np.ndarray, int],
) -> Tuple[np.ndarray, int, int]:
    """Wrapper that uses global arrays to avoid sending them each time.

    Parameters
    ----------
    args : tuple
        (i, j, intervals, max_lag)
        i : int
            First dimension index.
        j : int
            Second dimension index.
        intervals : (n_intervals, 2)
            Intervals array.
        max_lag : int
            Maximum lag.

    Returns
    -------
    result : tuple
        (cross_corr, i, j)
        cross_corr : (2 * max_lag + 1,)
            Cross-correlation values.
        i : int
            First dimension index.
        j : int
            Second dimension index.
    """
    global _c_global, _t_global
    i, j, intervals, max_lag = args
    # Your existing function – now uses globals instead of passing c, t
    cross_corr = _single_cross_correlation(
        _c_global, _t_global, i, j, intervals, max_lag
    )
    return cross_corr
