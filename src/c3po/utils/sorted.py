import numpy as np
import jax
import time
import yaml
from pathlib import Path
import pynwb
from pathlib import Path
import pandas as pd

from c3po.model.model import C3PO, train_model
from c3po.model.util import prep_training_data
from c3po.analysis.analysis import C3poAnalysis
from c3po.utils.model_configs import configs
from c3po.utils._defaults import DEFAULT_TRAIN_ARGS


def get_spike_data(nwbfile) -> tuple[np.ndarray, np.ndarray]:
    """Extract and time-sort spike events from all Units tables in an NWB file.

    Args:
        nwbfile: Loaded `pynwb.NWBFile` object containing one or more
            `pynwb.misc.Units` objects.

    Returns:
        tuple[np.ndarray, np.ndarray]: (mark_times, mark_ids), time-sorted arrays
            of spike times and the corresponding unit index for each spike.
    """
    units = []
    for obj in nwbfile.objects.values():
        if isinstance(obj, pynwb.misc.Units):
            units.append(obj.to_dataframe())

    spiking_df = pd.concat(units, ignore_index=True)

    mark_times = []
    mark_ids = []

    for i, times in enumerate(spiking_df["spike_times"]):
        mark_times.append(times)
        mark_ids.append(np.full_like(times, i, dtype=float))
    mark_times = np.concatenate(mark_times)
    mark_ids = np.concatenate(mark_ids)
    ind = np.argsort(mark_times)
    mark_times = mark_times[ind]
    mark_ids = mark_ids[ind]

    return mark_times, mark_ids


def get_sorted_c3po_model(
    n_units: int,
    dimensions: int,
    mua_rate: float,
    model_args: dict | Path | str = "sorted_default_v0",
) -> tuple:
    """Build a sorted-units C3PO model from a named/pre-built or on-disk config.

    Args:
        n_units (int): Number of sorted units (encoder input dimension).
        dimensions (int): Latent/context dimensionality of the model.
        mua_rate (float): Average multi-unit activity rate (Hz), used only for
            reporting context-window timescales.
        model_args (dict | Path | str, optional): Either a name of a built-in config
            in `c3po.utils.model_configs.configs`, a path to a JSON config file, or
            a dict of model arguments. Defaults to "sorted_default_v0".

    Returns:
        tuple: (model, model_args), the constructed `C3PO` model and the resolved
            model argument dictionary (with unit/dimension-specific entries filled in).
    """
    # Load the model_args from file if provided
    if isinstance(model_args, str) and model_args in configs:
        model_args = configs[model_args]

    elif isinstance(model_args, (Path, str)):
        from c3po.analysis.utils_io import read_config

        if not Path(model_args).exists():
            raise FileNotFoundError(f"Model args file {model_args} not found.")
        model_args = read_config(model_args)

    # assign model-specific parameters
    model_args["encoder_args"]["n_units"] = n_units
    model_args["latent_dim"] = dimensions
    model_args["context_dim"] = dimensions

    # report smoothing timescale if applicable
    smoothing = model_args["context_args"].get("smoothing", 0)
    if smoothing > 1:
        print(
            f"Smoothing is enabled with a value of {smoothing}, corresponding to a"
            f"timescale of {smoothing/mua_rate*1000:2f}ms."
        )
    else:
        print("Smoothing is disabled.")

    kernels = model_args["context_args"].get("layer_kernel_size", [])
    dilations = model_args["context_args"].get("layer_dilations", [])
    context_window = max([d * k for d, k in zip(dilations, kernels)])
    context_time = context_window / mua_rate
    print(
        f"Context window is {context_window} samples, corresponding to a timescale of {context_time*1000:.2f}ms."
    )
    model_args.update(
        {
            "predicted_sequence_length": 1,
            "return_embeddings_in_call": True,
        }
    )

    model = C3PO(**model_args)
    return model, model_args


def train_from_nwb(
    nwbfile: pynwb.NWBFile | str | Path,
    dimensions: int,
    output_dir: Path,
    sample_length: int = 2000,
    model_args: dict | Path | str = "sorted_default_v0",
    multi_gpu: bool = False,
    train_args: dict = dict(),
):
    """Train a sorted-units C3PO model from an in-memory or on-disk NWB file.

    Args:
        nwbfile (pynwb.NWBFile | str | Path): Loaded NWB file object, or a path to
            an NWB file to be read from disk.
        dimensions (int): Latent/context dimensionality of the model.
        output_dir (Path): Directory to save the trained model and training outputs.
        sample_length (int, optional): Number of spike events per training sample.
            Defaults to 2000.
        model_args (dict | Path | str, optional): Model configuration, see
            `get_sorted_c3po_model`. Defaults to "sorted_default_v0".
        multi_gpu (bool, optional): Currently unused. Defaults to False.
        train_args (dict, optional): Overrides for `DEFAULT_TRAIN_ARGS` passed to
            `train_model`. Defaults to an empty dict.

    Returns:
        The result of `train_from_sorted_data` (None; outputs are saved to `output_dir`).
    """
    # If nwbfile is a path, read the NWB file
    if isinstance(nwbfile, [str, Path]):
        import pynwb

        io = pynwb.NWBHDF5IO(nwbfile, "r")
        nwbfile = io.read()
    mark_times, mark_ids = get_spike_data(nwbfile)
    return train_from_sorted_data(
        mark_times,
        mark_ids,
        dimensions,
        output_dir,
        sample_length=sample_length,
        model_args=model_args,
        train_args=train_args
    )


def train_from_sorted_data(
    mark_times: np.ndarray,
    mark_ids: np.ndarray,
    dimensions: int,
    output_dir: Path,
    sample_length: int = 2000,
    model_args: dict | Path | str = "sorted_default_v0",
    train_args: dict = dict(),
    time_scale_unit: str = "ms",
) -> C3poAnalysis:
    """Train a sorted-units C3PO model directly from spike time/id arrays.

    Builds training windows, initializes and trains the model, embeds the full
    dataset, fits/interpolates context PCA, saves the resulting analysis object,
    and writes a diagnostic plot and training metrics to `output_dir`.

    Args:
        mark_times (np.ndarray): Spike times, shape (n_spikes,).
        mark_ids (np.ndarray): Unit index for each spike, shape (n_spikes,).
        dimensions (int): Latent/context dimensionality of the model.
        output_dir (Path): Directory to save the model, metrics, and plots.
        sample_length (int, optional): Number of spike events per training sample.
            Defaults to 2000.
        model_args (dict | Path | str, optional): Model configuration, see
            `get_sorted_c3po_model`. Defaults to "sorted_default_v0".
        train_args (dict, optional): Overrides for `DEFAULT_TRAIN_ARGS` passed to
            `train_model`. Defaults to an empty dict.
        time_scale_unit (str, optional): Units of `mark_times`; one of "ms", "us",
            or "s". Used to convert inter-spike intervals to milliseconds. Defaults
            to "ms".

    Returns:
        C3poAnalysis: The trained model/analysis outputs.
        Trained model/analysis outputs, a context PCA plot, and training
        metrics are saved to `output_dir`.
    """
    # Merge default train_args with user-provided train_args
    compiled_train_args = DEFAULT_TRAIN_ARGS.copy()
    compiled_train_args.update(train_args)

    # prep data
    delta_t = np.diff(mark_times)[None, ...]  # delay between each spike event
    if time_scale_unit == "ms":
        delta_t *= 1000
    elif time_scale_unit == "us":
        delta_t *= 1e6
    elif time_scale_unit != "s":
        raise ValueError("time_scale_unit must be 'ms', 'us', or 's'")
    x = mark_ids[1:][None, ...].astype(
        np.int16
    )  # unit id of each spike event, excluding the first event since it has no delta_t

    ind_valid = delta_t[0] > 0
    x = x[:, ind_valid]
    delta_t = delta_t[:, ind_valid]

    sample_length = sample_length  # number of spike events in each sample
    x_train, delta_t_train = prep_training_data(x, delta_t, sample_length)
    x_train = x_train[..., None]

    mua_rate = np.mean(delta_t) ** -1 * 1000
    print(f"Average multi-unit activity rate: {mua_rate:.2f} Hz")
    print(f"Average sample duration: {(sample_length/mua_rate):.2f} seconds")

    n_units = np.max(x) + 1
    model, model_args = get_sorted_c3po_model(
        n_units=n_units, dimensions=dimensions, mua_rate=mua_rate, model_args=model_args
    )
    print(f"Model initialized with {n_units} units and {dimensions} dimensions.")
    rand_key = jax.random.PRNGKey(0)
    params = model.init(
        jax.random.PRNGKey(0), x_train[:10], delta_t_train[:10], rand_key
    )
    training_start = time.time()
    params, tracked_loss = train_model(
        model,
        params,
        x_train,
        delta_t_train,
        **compiled_train_args
        # learning_rate=3e-4,  # if outputs end up as all zeros, try a smaller learning rate
        # n_epochs=1000,  # maximum number of epochs (usually will hit stop criteria long before this)
        # initial_batch_size=64,
        # buffer_size=8,
        # min_batch_size=32,
        # max_n_neg=128,
        # initial_n_neg=8,
        # multi_gpu=multi_gpu,  # If True and multiple GPUs are available, multiple batches can be processed in parallel
    )
    training_end = time.time()
    print(f"Training completed in {training_end-training_start:.2f} seconds.")
    # build c3po analysis object
    from c3po.analysis.analysis import C3poAnalysis

    model_args["n_neg_samples"] = (4,)  # no longer matters outside of training
    analysis = C3poAnalysis(
        model=model,
        model_args=model_args,
        params=params,
    )
    analysis.embed_data(
        x[..., None],
        delta_t,
        first_mark_time=mark_times[0],
        chunk_size=5000,
        delta_t_units=time_scale_unit,
    )
    analysis.fit_context_pca()
    t_interp = np.arange(analysis.t[0], analysis.t[-1], 0.001)
    analysis.interpolate_context(t_interp)
    analysis.embed_context_pca()
    # Save to disc
    analysis.save_model(output_dir)

    # Plot example traces
    import matplotlib.pyplot as plt

    ind_plot = slice(10000, 10000 + 5000)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    plt.plot(analysis.t[ind_plot], analysis.c_pca[ind_plot][:, :3])
    plt.xlabel("Time (s)")
    plt.ylabel("Context Data")
    plt.title("(Reloaded) PCA-Transformed Context Data Over Time")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_dir / "context_pca_plot.png", dpi=300)

    metrics = {
        "training_time": float(training_end - training_start),
        "n_units": int(n_units),
        "dimensions": int(dimensions),
        "mua_rate": float(mua_rate),
        "time_scale_unit": time_scale_unit,
    }
    # Save metrics as a yaml file
    with open(output_dir / "metrics.yaml", "w") as f:
        yaml.dump(metrics, f)
    return analysis