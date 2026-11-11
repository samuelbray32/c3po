from typing import Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn

from .util import MLP


class BaseEncoder(nn.Module):
    """Base class for mark encoders, mapping a single mark to a latent vector.

    Encoders are applied per time step (vmapped over the marks axis) by `Embedding`.

    Parameters
    ----------
    latent_dim : int
        Dimension of the output latent space (Z).
    """

    latent_dim: int

    def setup(self):
        raise NotImplementedError

    def __call__(self, x):
        """Encode marks `x` of shape (n_batch, input_dim) to (n_batch, latent_dim)."""
        raise NotImplementedError

    @property
    def requires_random_key(self) -> bool:
        """Indicates whether the encoder requires a random key for its operations."""
        return False

def encoder_factory(encoder_model: str, latent_dim: int, **kwargs) -> BaseEncoder:
    """Construct an encoder by name.

    Parameters
    ----------
    encoder_model : str
        Key of `model_dict` (e.g. "simple", "convolutional1D", "multi_shank_v0",
        "multi_shank_v1", "sorted_spikes", "identity", "hybrid").
    latent_dim : int
        Dimension of the latent space (Z).
    **kwargs
        Additional arguments for the chosen encoder class.

    Returns
    -------
    BaseEncoder
    """
    if encoder_model in model_dict:
        return model_dict[encoder_model](latent_dim=latent_dim, **kwargs)
    else:
        raise ValueError(f"Unknown encoder model: {encoder_model}")


class SimpleEncoder(BaseEncoder):
    """MLP encoder with hidden layer sizes `widths` and a `latent_dim` output layer."""

    widths: Sequence[int]

    def setup(self):
        self.encoder = MLP(
            self.widths + (self.latent_dim,), kernel_init=nn.initializers.he_uniform()
        )

    def __call__(self, x):
        return self.encoder(x)


class convolutionalEncoder1D(BaseEncoder):
    """1D convolutional layers over the input features followed by an MLP.

    `conv_kernel_sizes`, `conv_strides` and `conv_features` must have equal length;
    `widths` are the hidden sizes of the final MLP (output size `latent_dim`).
    """

    conv_kernel_sizes: Sequence[int]
    conv_strides: Sequence[int]
    conv_features: Sequence[int]
    widths: Sequence[int]

    def setup(self):
        if (
            not len(self.conv_kernel_sizes)
            == len(self.conv_strides)
            == len(self.conv_features)
        ):
            raise ValueError(
                "conv_kernel_sizes and conv_strides must have the same length"
            )
        conv_layers = []
        print("HI")
        for i, (kernel_size, stride, features) in enumerate(
            zip(self.conv_kernel_sizes, self.conv_strides, self.conv_features)
        ):
            # if i == 0:
            #     kernel = kernel_size
            # else:
            #     kernel = 1
            print(i)
            conv_layers.append(nn.Conv(features, kernel_size, stride, padding="VALID"))
            # self.conv_layers.append(nn.relu)
        self.conv_layers = conv_layers
        print("HI")
        # self.flatten = nn.Flatten()
        self.dense = MLP(
            self.widths + (self.latent_dim,), kernel_init=nn.initializers.he_uniform()
        )

    def __call__(self, x):
        x = jnp.expand_dims(x, axis=-1)
        for layer in self.conv_layers:
            print(x.shape)
            x = layer(x)

        print(x.shape)
        x = jax.lax.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
        return self.dense(x)


class MultiShankEncoderV0(BaseEncoder):
    """makes a separate encoder model for each shank then combines

    Input has shape (n_batch, n_channels, n_shanks); the encoding of each shank is
    summed over the shanks with non-zero data. `shank_encoder_params` are the
    `encoder_factory` arguments (including `encoder_model`) for each shank encoder.
    """

    n_shanks: int
    shank_encoder_params: dict
    latent_dim: int

    def setup(self):
        self.shank_encoders = [
            encoder_factory(**self.shank_encoder_params, latent_dim=self.latent_dim)
            for _ in range(self.n_shanks)
        ]

    def __call__(self, x):
        print("x shape", x.shape)
        print("x shape", x[..., 0].shape)
        encoded_shanks = [
            encoder(x[..., i]) for i, encoder in enumerate(self.shank_encoders)
        ]
        print("encoded shanks shape", encoded_shanks[0].shape)
        encoded_shanks = jnp.stack(encoded_shanks, axis=-1)
        print("encoded shanks shape", encoded_shanks.shape)
        shank_indicator = jnp.sum(jnp.abs(x), axis=-2)
        print("shank indicator shape", shank_indicator.shape)
        shank_indicator = jnp.where(shank_indicator != 0, 1.0, 0.0)
        print("shank indicator shape", shank_indicator.shape)
        encoded_shanks = jnp.sum(encoded_shanks * shank_indicator[:, None, :], axis=-1)
        # encoded_shanks = jnp.sum(encoded_shanks, axis=-1)
        print("encoded shanks shape", encoded_shanks.shape)
        return encoded_shanks


class MultiShankEncoderV1(BaseEncoder):
    """makes a separate encoder model for each shank then combines
    assumes that the shank indicator is the last entry in the input vector

    Input has shape (n_batch, n_channels+1). Each mark is encoded by the encoder
    of its shank; `shank_encoder_params` are the `encoder_factory` arguments
    (including `encoder_model`) shared by the `n_shanks` encoders.
    """

    n_shanks: int
    shank_encoder_params: dict
    latent_dim: int

    def setup(self):
        self.shank_encoders = [
            encoder_factory(**self.shank_encoder_params, latent_dim=self.latent_dim)
            for _ in range(self.n_shanks)
        ]

        self.expanded_shank_encoders = [
            lambda dummmy_self, x: encoder(x) for encoder in self.shank_encoders
        ]

    def __call__(self, x):
        # x has shape (batch_size, n_channels+1)
        print("x shape", x.shape)
        # Separate the data part (all but last channel) from the integer shank indicator
        x_data = x[..., :-1]  # shape: (batch_size, n_channels)
        x_shank = x[..., -1].astype(int)  # shape: (batch_size,)
        batch_size, _ = x_data.shape

        # Prepare an output array to store latent vectors of shape (batch_size, latent_dim)
        outputs = jnp.zeros(
            (batch_size + 1, self.latent_dim), dtype=x_data.dtype
        )  # last index is a garbage slot

        # For each shank index, gather the data belonging to that shank, encode, and scatter back
        for shank_idx in range(self.n_shanks):
            # Gather integer indices where x_shank == shank_idx
            # 'size=batch_size' ensures a fixed-size output for JIT
            idxs = jnp.where(x_shank == shank_idx, size=batch_size, fill_value=-1)[0]

            # Select the corresponding data rows
            data_for_shank = x_data[idxs]  # shape: (num_selected, n_channels)

            # Encode using the shank-specific encoder
            encoded_data = self.shank_encoders[shank_idx](
                data_for_shank
            )  # (num_selected, latent_dim)

            # Scatter the encoded data back
            outputs = outputs.at[idxs].set(encoded_data)
        return outputs[:-1]

class HybridEncoder(BaseEncoder):
    """makes a separate encoder model for clustered and unclustered spikes then combines

    Data is assumed to be of shape (batch_size, n_channels+1)
    The last channel is the sort indicator:
        - integer >= 0: the mark is sorted; this value is passed to `sorted_encoder`
        - -1: the mark is unsorted (clusterless); the data channels are passed to
          `clusterless_encoder`

    `sorted_encoder_params` and `clusterless_encoder_params` are the
    `encoder_factory` arguments (including `encoder_model`) for each encoder.

    """

    latent_dim: int
    sorted_encoder_params: dict
    clusterless_encoder_params: dict

    def setup(self):
        self.sorted_encoder = encoder_factory(**self.sorted_encoder_params, latent_dim=self.latent_dim)
        self.clusterless_encoder = encoder_factory(**self.clusterless_encoder_params, latent_dim=self.latent_dim)

    def __call__(self, x):
        # x has shape (batch_size, n_channels+1)
        print("x shape", x.shape)
        # Separate the data part (all but last channel) from the integer shank indicator
        x_data = x[..., :-1]  # shape: (batch_size, n_channels+1)
        x_sort_id = x[..., -1].astype(int)  # shape: (batch_size,)
        x_sort_id = x_sort_id[..., None]  # shape: (batch_size, 1)

        batch_size, _ = x_data.shape

        # Prepare an output array to store latent vectors of shape (batch_size, latent_dim)
        outputs = jnp.zeros(
            (batch_size + 1, self.latent_dim), dtype=x_data.dtype
        )  # last index is a garbage slot

        # embed the sorted spikes
        ind_sorted = jnp.where(x_sort_id >= 0, size=batch_size, fill_value=-1)[0]
        data_sorted = x_sort_id[ind_sorted]
        encoded_sorted = self.sorted_encoder(data_sorted)
        outputs = outputs.at[ind_sorted].set(encoded_sorted)

        # embed the clusterless spikes
        ind_clusterless = jnp.where(x_sort_id == -1, size=batch_size, fill_value=-1)[0]
        data_clusterless = x_data[ind_clusterless,:-1]
        encoded_clusterless = self.clusterless_encoder(data_clusterless)
        outputs = outputs.at[ind_clusterless].set(encoded_clusterless)

        return outputs[:-1]



class SortedSpikesEncoder(BaseEncoder):
    """Embeds sorted spike data by a learned per-unit latent vector.

    Parameters
    ----------
    n_units : int
        Number of sorted units.
    gauss_noise_std : float
        If > 0, standard normal noise is added to the latents and a random key
        is required (`requires_random_key`). The value is used only as an on/off
        flag; the noise scale is not set by it.
    input_format : str
        "one_hot" for inputs of shape (n_batch, n_units), or "indices" for
        inputs of shape (n_batch, 1) holding unit indices.
    """

    n_units: int
    gauss_noise_std: float = 0.0
    input_format: str = "one_hot"  # or "indices"

    def setup(self):
        self.m = self.param(
            "encoder_matrix",
            # nn.initializers.he_uniform(),
            nn.initializers.normal(stddev=0.1),
            (self.n_units, self.latent_dim),
        )

    def __call__(self, x, rand_key=None):
        """Encode `x` to latents of shape (n_batch, latent_dim)."""
        if self.input_format == "one_hot":
            if x.shape[-1] != self.n_units:
                raise ValueError(
                    f"Input shape {x.shape} does not match n_units {self.n_units}"
                )
            projection = jnp.dot(x, self.m)
        elif self.input_format == "indices":
            projection = self.m[x][:, 0, :]  # x is of shape (batch_size, 1)
        else:
            raise ValueError(
                f"Unknown input_format {self.input_format}. Use 'one_hot' or 'indices'"
            )

        if not self.requires_random_key:
            return projection
        return jax.random.normal(rand_key, (x.shape[0], self.latent_dim)) + projection

    @property
    def requires_random_key(self):
        """Indicates whether the encoder requires a random key for its operations."""
        return self.gauss_noise_std > 0


class IdentityEncoder(BaseEncoder):
    """Returns the input unchanged; the input dimension must equal `latent_dim`."""

    def setup(self):
        return

    def __call__(self, x):
        if x.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Input shape {x.shape} does not match latent_dim {self.latent_dim}"
            )
        # Simply return the input as is
        return jnp.array(x)


model_dict = {
    "simple": SimpleEncoder,
    "convolutional1D": convolutionalEncoder1D,
    "multi_shank_v0": MultiShankEncoderV0,
    "multi_shank_v1": MultiShankEncoderV1,
    "sorted_spikes": SortedSpikesEncoder,
    "identity": IdentityEncoder,
    "hybrid": HybridEncoder,
}