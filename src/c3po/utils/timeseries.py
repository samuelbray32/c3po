"""Utilities for smoothing 1D/2D time series data."""

import numpy as np
import scipy.signal


def smooth(
    data: np.ndarray, n: int = 5, sigma: float = None, hamming: bool = False
) -> np.ndarray:
    """Smooth a 1D or 2D array along its first axis with a kernel of size `n`.

    Args:
        data (np.ndarray): Array to smooth, shape (n_samples,) or (n_samples, n_features).
        n (int, optional): Kernel length. Rounded up to the nearest odd number. Defaults to 5.
        sigma (float, optional): Standard deviation of the gaussian kernel. Ignored if
            `hamming` is True. Defaults to `n / 2` if not given.
        hamming (bool, optional): If True, use a uniform (boxcar) kernel of length `sigma`
            instead of a gaussian kernel. Defaults to False.

    Returns:
        np.ndarray: Smoothed array, same shape as `data`.
    """
    if n % 2 == 0:
        n += 1  # make sure n is odd
    if sigma is None:
        sigma = n / 2
    kernel = gkern(n, sigma)[:, None]
    if hamming:
        n = sigma
        kernel = np.ones((sigma, 1)) / sigma
    if len(data.shape) == 1:
        pad = np.ones(((n - 1) // 2, 1))
        return np.squeeze(
            scipy.signal.convolve2d(
                np.concatenate(
                    [pad * data[:, None][0], data[:, None], pad * data[:, None][-1]],
                    axis=0,
                ),
                kernel,
                mode="valid",
            )
        )
    else:
        pad = np.ones(((n - 1) // 2, data.shape[1]))
        return scipy.signal.convolve2d(
            np.concatenate([pad * data[0], data, pad * data[-1]], axis=0),
            kernel,
            mode="valid",
        )


def gkern(l: int = 5, sig: float = 1.0) -> np.ndarray:
    """Create a 1D gaussian kernel.

    Args:
        l (int, optional): Side length of the kernel. Defaults to 5.
        sig (float, optional): Standard deviation of the gaussian. Defaults to 1.0.

    Returns:
        np.ndarray: Normalized 1D gaussian kernel of length `l` (sums to 1).
    """
    ax = np.linspace(-(l - 1) / 2.0, (l - 1) / 2.0, l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    return gauss / np.sum(gauss)
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)
