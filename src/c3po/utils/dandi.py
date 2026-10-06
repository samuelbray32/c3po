from pathlib import Path
from dandi.consts import known_instances
from dandi.dandiapi import DandiAPIClient
from dandi.consts import known_instances
import fsspec
from fsspec.implementations.cached import CachingFileSystem
import pynwb
from pathlib import Path
import h5py
import pandas as pd
import numpy as np
import jax
from c3po.model.model import C3PO, train_model
from c3po.model.util import prep_training_data


# Helpers
def get_dandi_file(dandiset_id, dandi_path, dandi_instance="dandi"):
    tmp_dir = "dandi_cache"  # Local folder for cache
    Path(tmp_dir).mkdir(exist_ok=True)  # Create the cache folder if it doesn't exist

    # get the s3 url from Dandi
    with DandiAPIClient(
        dandi_instance=known_instances[dandi_instance],
    ) as client:
        asset = client.get_dandiset(dandiset_id).get_asset_by_path(dandi_path)
        s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

    # stream the file from s3
    # first, create a virtual filesystem based on the http protocol
    fs = fsspec.filesystem("http")

    # create a cache to save downloaded data to disk (optional)
    fsspec_file = CachingFileSystem(
        fs=fs,
        cache_storage=tmp_dir,  # Local folder for cache
    )

    # Open and return the file
    fs_file = fsspec_file.open(s3_url, "rb")
    io = pynwb.NWBHDF5IO(file=h5py.File(fs_file))
    nwbfile = io.read()
    return io, nwbfile


def get_spike_data(nwbfile):
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


def get_default_c3po_model(n_units, dimensions, mua_rate):
    # Dimensionality, recommend to be same for both latent and context spaces
    latent_dim = dimensions
    context_dim = dimensions

    # Encoder model (waveform (neuron_id) to Z)
    encoder_args = dict(
        encoder_model="sorted_spikes",
        n_units=n_units,
        input_format="indices",
    )

    # Context model (Z to C)
    dilations = [
        1,
        2,
        4,
        8,
        16,
    ]
    kernels = [8, 8, 16, 16, 32]
    dilations = dilations * 2
    kernels = kernels * 2
    smoothing = 10
    context_args = dict(
        context_model="wavenet",
        layer_dilations=dilations,
        layer_kernel_size=kernels,
        expanded_dim=64,
        smoothing=smoothing,
        smoothing_decay=0.9,
        categorical=False,
    )
    if smoothing > 1:
        print(
            f"Smoothing is enabled with a value of {smoothing}, corresponding to a"
            f"timescale of {smoothing/mua_rate*1000:2f}ms."
        )
    else:
        print("Smoothing is disabled.")

    context_window = max([d * k for d, k in zip(dilations, kernels)])
    context_time = context_window / mua_rate
    print(
        f"Context window is {context_window} samples, corresponding to a timescale of {context_time*1000:.2f}ms."
    )

    # Rate model (C to hazard function parameters)
    # Recommend to NOT change for best interpretability
    rate_args = dict(
        rate_model="sharedSpace",
    )

    # Process model of the conditional Hazard function
    # Other options implemented but never found a case where necessary in dense
    # recordings with high mua firing rates
    process_model = "poisson"

    model_args = dict(
        encoder_args=encoder_args,
        context_args=context_args,
        rate_args=rate_args,
        distribution=process_model,
        latent_dim=latent_dim,
        context_dim=context_dim,
    )
    model = C3PO(
        **model_args,
        n_neg_samples=8,
        predicted_sequence_length=1,
        return_embeddings_in_call=True,
    )
    return model, model_args


def train_from_dandi(dandiset_id, dandi_path, dandi_instance, dimensions, output_dir):
    import time

    io, nwbfile = get_dandi_file(dandiset_id, dandi_path, dandi_instance)
    mark_times, mark_ids = get_spike_data(nwbfile)

    scale_unit = "ms"
    delta_t = np.diff(mark_times)[None, ...]  # delay between each spike event
    if scale_unit == "ms":
        delta_t *= 1000
    x = mark_ids[1:][None, ...].astype(
        np.int16
    )  # unit id of each spike event, excluding the first event since it has no delta_t

    ind_valid = delta_t[0] > 0
    x = x[:, ind_valid]
    delta_t = delta_t[:, ind_valid]

    sample_length = 2000  # number of spike events in each sample
    x_train, delta_t_train = prep_training_data(x, delta_t, sample_length)
    x_train = x_train[..., None]

    mua_rate = np.mean(delta_t) ** -1 * 1000
    print(f"Average multi-unit activity rate: {mua_rate:.2f} Hz")
    print(f"Average sample duration: {(sample_length/mua_rate):.2f} seconds")

    n_units = np.max(x) + 1
    model, model_args = get_default_c3po_model(
        n_units=n_units, dimensions=dimensions, mua_rate=mua_rate
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
        learning_rate=3e-4,  # if outputs end up as all zeros, try a smaller learning rate
        n_epochs=1000,  # maximum number of epochs (usually will hit stop criteria long before this)
        initial_batch_size=64,
        buffer_size=8,
        min_batch_size=32,
        max_n_neg=128,
        initial_n_neg=8,
        multi_gpu=False,  # If True and multiple GPUs are available, multiple batches can be processed in parallel
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
        delta_t_units=scale_unit,
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
    }
    # Save metrics as a yaml file
    import yaml

    with open(output_dir / "metrics.yaml", "w") as f:
        yaml.dump(metrics, f)
