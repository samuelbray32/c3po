from typing import Sequence

import flax.linen as nn
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

def prep_training_data(x: np.ndarray, delta_t: np.ndarray, sample_length:int = 2000,
                       overlap_fraction: float = 0.5, valid_indices = None):
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