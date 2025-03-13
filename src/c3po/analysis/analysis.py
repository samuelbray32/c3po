import numpy as np
from spyglass.common import interval_list_contains_ind
from sklearn.decomposition import PCA

import jax

from ..model.model import Embedding


class C3poAnalysis:
    def __init__(
        self,
        model,
        model_args,
        params,
    ):
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
        if not all([(val is None) for val in [self.z, self.c, self.t]]):
            Warning("Clearing existing data")
            self.z = None
            self.c = None
            self.t = None
            self.t_interp = None
            self.c_interp = None
            self.c_pca = None
            self.pca_intervals = None
            self.c_pca_interp = None

        z = np.zeros((x.shape[1], self.latent_dim))
        c = np.zeros((x.shape[1], self.context_dim))

        @jax.jit
        def embed_chunk(x, delta_t):
            return Embedding(
                self.encoder_args, self.context_args, self.latent_dim, self.context_dim
            ).apply(self.embedding_params, x, delta_t)

        i = chunk_padding
        from tqdm import tqdm

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

        self.z = np.array(z)
        self.c = np.array(c)

        # store the markls times in unix time
        t = np.cumsum(delta_t)
        if delta_t_units == "ms":
            t *= 1e-3
        elif delta_t_units not in ["s", "ms"]:
            raise ValueError(f"delta_t_units must be 's' or 'ms' not {delta_t_units}")
        self.t = t + first_mark_time

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
        pca = PCA(n_components=self.context_dim)
        pca.fit(self.c[fit_ind])
        self.pca = pca
        self.pca_intervals = fit_intervals
        self.embed_context_pca()

    def embed_context_pca(self):
        if self.pca is None:
            raise ValueError("pca must first be fit to data")

        if self.c_pca is None:
            self.c_pca = self.pca.transform(self.c)
        if self.c_interp is not None and self.c_pca_interp is None:
            self.c_pca_interp = self.pca.transform(self.c_interp)

    # ----------------------------------------------------------------------------------
    # Latent factor interpretation tools

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
        if pca and interpolated:
            t_data = self.t_interp
            c_data = self.c_pca_interp
        elif pca and not interpolated:
            t_data = self.t
            c_data = self.c
        elif not pca and interpolated:
            t_data = self.t_interp
            c_data = self.c_interp
        elif not pca and not interpolated:
            t_data = self.t
            c_data = self.c_interp

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
        c = self.c_pca if pca else self.c
        for i in range(len(bins) - 1):
            ind_bin = np.logical_and(
                context_features > bins[i], context_features < bins[i + 1]
            )
            context_binned.append(c_data[ind_context][ind_bin])
        return context_binned, bins

        # context_binned = []
        # for i in range(len(bins) - 1):
        #     ind_bin = np.logical_and(
        #         feature[ind_feature] > bins[i], feature[ind_feature] < bins[i + 1]
        #     )
        #     context_binned.append(self.c[ind_context][ind_bin])
        # return context_binned

    def alligned_response(
        self, marks: np.ndarray, t_plot: tuple[float, float], pca=True
    ):
        """
        Look at the model "response function" around the mark times

        Args:
            marks (np.ndarray): Time points to allign to (in seconds) (ex. stimulus, reward, neuron firing).
            t_plot (tuple(float,float)): Time window to plot around each mark time (ex. (-.1,.5)).
        """
        # interpolate the context if not already done
        try:
            self._check_interpolated_data()
        except ValueError:
            self.interpolate_context()
            self.embed_context_pca()

        c_data = self.c_pca_interp if pca else c_interp

        dt = np.median(np.diff(self.t_interp))
        window = int(t_plot[0] / dt), int(t_plot[1] / dt)
        mark_inds = np.digitize(marks, self.t_interp) - 1
        mark_inds = mark_inds[
            np.logical_and(
                mark_inds + window[0] >= 0, mark_inds + window[1] < c_data.shape[0]
            )
        ]
        response = []
        for i in mark_inds:
            response.append(c_data[i + window[0] : i + window[1]])
        return np.array(response)

    def _check_embedded_data(self):
        if any(val is None for val in [self.z, self.c, self.t]):
            raise ValueError("Data not embedded yet")

    def _check_interpolated_data(self):
        self._check_embedded_data()
        if any(val is None for val in [self.t_interp, self.c_interp]):
            raise ValueError("Data not interpolated yet")
