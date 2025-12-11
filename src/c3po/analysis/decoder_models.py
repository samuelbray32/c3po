from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
import numpy as np


class DiscretizedRegression:
    def __init__(
        self,
        n_bins=10,
        bin_strategy="uniform",
        max_iter=1000,
        balance_groups=False,
        multidim=False,
    ):
        if multidim:
            self.discretizer = Ordinal2dDiscretizer(
                n_bins=n_bins, bin_strategy=bin_strategy
            )
        else:
            self.discretizer = KBinsDiscretizer(
                n_bins=n_bins, encode="ordinal", strategy=bin_strategy
            )
        self.multidim = multidim
        self.model = LogisticRegression(max_iter=max_iter)
        self.balance_groups = balance_groups

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
        if self.balance_groups:
            class_counts = np.bincount(y_binned.astype(int).flatten())
            total_counts = len(y_binned)
            class_weights = {
                i: total_counts / (len(class_counts) * count)
                for i, count in enumerate(class_counts)
                if count > 0
            }
            self.model = LogisticRegression(
                max_iter=self.model.max_iter, class_weight=class_weights
            )

        self.model.fit(X, y_binned)

    def predict(self, X):
        # y_binned_pred = self.model.predict(X)
        # return self.discretizer.inverse_transform(y_binned_pred[:, None])
        y_binned_probs = self.model.predict_proba(X)
        if self.multidim:
            ordinal_pred = self.model.predict(X)
            return self.discretizer.inverse_transform(ordinal_pred[:, None])

        bin_centers = (
            self.discretizer.bin_edges_[0][:-1] + self.discretizer.bin_edges_[0][1:]
        ) / 2
        y_pred = np.dot(y_binned_probs, bin_centers)
        return y_pred[:, None]


class Ordinal2dDiscretizer:
    def __init__(self, n_bins=10, bin_strategy="uniform"):
        self.n_bins = n_bins
        self.bin_strategy = bin_strategy
        self.discretizers = [
            KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=bin_strategy)
            for _ in range(2)
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
        y_ordinal = y_binned[:, 0] * self.n_bins + y_binned[:, 1]
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

        y_ordinal = y_binned[:, 0] * self.n_bins + y_binned[:, 1]
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
        y_binned[:, 0] = y_ordinal[:, 0] // self.n_bins
        y_binned[:, 1] = y_ordinal[:, 0] % self.n_bins

        y = np.zeros_like(y_binned)
        for i in range(2):
            y[:, i : i + 1] = self.discretizers[i].inverse_transform(
                y_binned[:, i : i + 1]
            )
        return y
