from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp
import jax
import numpy as np


class MLP(nn.Module):
    """Simple MLP with leaky relu activations. Used in custom models."""

    features: Sequence[int]
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.leaky_relu(nn.Dense(feat, kernel_init=self.kernel_init)(x))
        x = nn.Dense(self.features[-1], kernel_init=self.kernel_init)(x)
        return x


def prep_training_data(
    x: np.ndarray,
    delta_t: np.ndarray,
    sample_length: int = 2000,
    overlap_fraction: float = 0.5,
    valid_indices=None,
):
    """Prepares training data for the model.

    Args:
        x (np.ndarray): Sequence of marks. Shape (n_samples).
        delta_t (np.ndarray): Sequence of time intervals. Shape (n_samples).
        sample_length (int, optional): length of each training sample. Defaults to 2000.
        overlap_fraction (float, optional): overlap in time of pairs of training data. Defaults to 0.5.
        valid_indices (_type_, optional): indices to include in traing. Defaults of None includes all times.
    """
    i = sample_length
    x_train = []
    delta_t_train = []
    while i < x.shape[1]:
        x_train.append(x[0, i - sample_length : i])
        delta_t_train.append(delta_t[0, i - sample_length : i])
        i += int(sample_length * (1 - overlap_fraction))
        print(i)
    x_train = np.array(x_train)
    delta_t_train = np.array(delta_t_train)

    return x_train, delta_t_train


class StabilizedDenseDynamics(nn.Module):
    """Dense layer with stabilized linear dynamics (max eigenvalue <=1).
    Used in custom models."""

    features: int
    kernel_init: nn.initializers.Initializer = (
        None  # = nn.initializers.xavier_uniform()
    )

    bias_init: nn.initializers.Initializer = nn.initializers.zeros
    delta_matrix: bool = True

    def setup(self):
        self.kernel = self.param(
            "kernel", self.kernel_init, (self.features, self.features)
        )
        self.bias = self.param("bias", self.bias_init, (self.features,))

    def stable_matrix(self):
        return stabilize_matrix(self.kernel, self.delta_matrix)

    def __call__(self, x, stabilized_kernel=None):
        # stabilized_kernel = None
        if stabilized_kernel is None:
            stabilized_kernel = stabilize_matrix(self.kernel, self.delta_matrix)
        if self.delta_matrix:
            new_x = x - jnp.dot(x, stabilized_kernel)  # + self.bias
        else:
            new_x = jnp.dot(x, stabilized_kernel)  # + self.bias
        return new_x  # , stabilized_kernel


def stabilize_matrix(matrix, delta_matrix=False):
    """Stabilizes a matrix by scaling it to have the largest eigenvalue <= 1.

    Args:
        matrix (jax.numpy.ndarray): Matrix to be stabilized.
        delta_matrix (bool, optional): Whether the matrix is a delta matrix. Defaults to False.
        ex) delta_matrix = True X_t = X_{t-1} - (M)X_t
            delta_matrix = False X_t = (M)X_t
    """
    # return matrix
    if delta_matrix:
        matrix = jnp.eye(matrix.shape[0]) - matrix
    # get the larest eigenvalue of the matrix
    # eigvals = jnp.linalg.eigvals(matrix)
    # eigval_max = jnp.abs(jnp.max(eigvals))
    eigval_max = estimate_largest_eigenvalue(matrix, num_iterations=10)
    # scale the matrix to have the largest eigenvalue less than 1
    stabilized_matrix = matrix / jnp.maximum(1.0, eigval_max)
    if delta_matrix:
        stabilized_matrix = stabilized_matrix - jnp.eye(matrix.shape[0])
    return stabilized_matrix


def estimate_largest_eigenvalue(matrix, num_iterations=100, tol=1e-6):
    """
    Estimates the largest eigenvalue of a matrix using the power iteration method.

    Args:
        matrix (jax.numpy.ndarray): Square matrix whose largest eigenvalue is to be estimated.
        num_iterations (int): Maximum number of iterations for power iteration.
        tol (float): Convergence tolerance for eigenvalue difference.

    Returns:
        float: Estimated largest eigenvalue.
    """
    n = matrix.shape[0]
    # Randomly initialize the eigenvector
    v = jax.random.normal(jax.random.PRNGKey(0), (n,))
    v = v / jnp.linalg.norm(v)

    largest_eigenvalue = 0.0

    for _ in range(num_iterations):
        # Apply the matrix to the vector
        v_new = jnp.dot(matrix, v)
        v_new_norm = jnp.linalg.norm(v_new)
        v_new = v_new / v_new_norm

        # Estimate the eigenvalue
        new_eigenvalue = jnp.dot(v_new, jnp.dot(matrix, v_new))

        # Check for convergence
        # if jnp.abs(new_eigenvalue - largest_eigenvalue) < tol:
        #     break

        largest_eigenvalue = new_eigenvalue
        v = v_new

    return largest_eigenvalue


class CausalConv1D(nn.Module):
    """1D Causal Convolutional Layer."""

    features: int  # Number of output channels
    kernel_size: int  # Size of the convolutional kernel
    stride: int = 1  # Stride for the convolution
    use_bias: bool = False  # Whether to include a bias term

    def setup(self):
        self.conv = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            use_bias=self.use_bias,
            padding="VALID",  # No automatic padding; we'll do it manually
        )

    def __call__(self, x):
        """
        Apply causal convolution.

        Args:
            x: Input tensor of shape (batch, time, channels)

        Returns:
            Output tensor of shape (batch, time, features)
        """
        # Compute the required padding (only left padding)
        pad_width = self.kernel_size - 1  # Causal padding amount

        # Pad input with zeros on the left
        x_padded = jnp.pad(x, ((0, 0), (pad_width, 0), (0, 0)), mode="constant")

        # Apply convolution
        return self.conv(x_padded)


class DilatedCausalConv1D(CausalConv1D):
    """Causal Convolution with Dilated Filters."""

    dilation: int = 1  # Dilation factor

    def setup(self):
        self.conv = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            use_bias=self.use_bias,
            padding="VALID",  # No automatic padding; we'll do it manually
            kernel_dilation=(self.dilation,),
        )

    def __call__(self, x):
        pad_width = int((self.kernel_size - 1) * self.dilation)
        x_padded = jnp.pad(x, ((0, 0), (pad_width, 0), (0, 0)), mode="mean")
        return self.conv(x_padded)


# class Wavenet(nn.Module):
#     """Wavenet Model.
#     Citation: https://arxiv.org/pdf/1609.03499
#     """
#     layer_dilations: Sequence[int]
#     layer_kernel_size: Sequence[int]
#     layer_features: Sequence[int]
#     num_channels: int # Number of output channels

#     def setup(self):
#         self.layers = [
#             DilatedCausalConv1D(
#                 features=self.layer_features[i],
#                 kernel_size=self.layer_kernel_size[i],
#                 dilation=self.layer_dilations[i],
#             )
#             for i in range(len(self.layer_dilations))
#         ]
#         self.gating_layers = [
#             DilatedCausalConv1D(
#                 features=self.layer_features[i],
#                 kernel_size=self.layer_kernel_size[i],
#                 dilation=self.layer_dilations[i],
#             )
#             for i in range(len(self.layer_dilations))
#         ]
#         self.final_conv = CausalConv1D(features=self.num_channels, kernel_size=1)

#     def __call__(self, x):
#         for layer in self.layers:
#             x = jnp.tanh(layer(x)) * jax.nn.sigmoid(self.gating_layers(x)) # Gated activation units
#         return self.final_conv(x)

from functools import partial


def causal_smoothing(x, filter_size=10):
    pad_width = filter_size - 1  # Causal padding amount
    # Pad input with zeros on the left
    x_padded = jnp.pad(x, ((0, 0), (pad_width, 0), (0, 0)), mode="constant")

    # window = jnp.ones((filter_size, x.shape[-1])) / filter_size
    # x = jax.lax.conv_general_dilated(x_padded, window, padding="VALID",window_strides=1)

    window = jnp.ones((filter_size,)) / filter_size
    conv = lambda y: jnp.convolve(y, window, mode="valid")
    x = jax.vmap(jax.vmap(conv, in_axes=-1, out_axes=-1), in_axes=0, out_axes=0)(
        x_padded
    )
    return x  # [:, pad_width:]
