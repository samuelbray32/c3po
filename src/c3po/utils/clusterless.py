"""Train C3PO models from detected spikes and continuous waveform features.

Features must already be extracted and scaled appropriately; spike detection and
feature extraction are not performed here.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import matplotlib.pyplot as plt
import numpy as np
import yaml

import logging
logger = logging.getLogger(__name__)

from c3po.analysis.analysis import C3poAnalysis
from c3po.analysis.utils_io import read_config
from c3po.model.model import C3PO, train_model
from c3po.model.util import prep_training_data
from c3po.utils.model_configs import configs
from c3po.utils._defaults import DEFAULT_TRAIN_ARGS

__all__ = ["train_from_waveform_data"]



def _resolve_clusterless_model_args(
    model_args: dict[str, Any] | Path | str = "clusterless_multishank_default_v0",
    dimensions: int = 16,
    n_neg_samples: int = 8,
    n_shanks: int | None = None,
) -> dict[str, Any]:
    """Copy a model configuration and configure its continuous-feature encoder."""
    if isinstance(model_args, str) and model_args in configs:
        resolved = deepcopy(configs[model_args])
    elif isinstance(model_args, (str, Path)):
        resolved = read_config(model_args)
    elif isinstance(model_args, dict):
        resolved = deepcopy(model_args)
    else:
        raise TypeError("model_args must be a dictionary, config name, or path")

    encoder_args = resolved["encoder_args"]
    if encoder_args["encoder_model"] in {"sorted_spikes", "hybrid"}:
        raise ValueError("model_args must specify a continuous waveform encoder.")

    if n_shanks is not None and n_shanks > 1:
        if encoder_args["encoder_model"] == "multi_shank_v1":
            encoder_args["n_shanks"] = n_shanks
        elif encoder_args["encoder_model"] in {"simple", "convolutional1D"}:
            logger.warning(
                "shank_ids supplied but encoder is not multi_shank_v1; "
                "switching to multi_shank_v1 with the same shank encoder."
            )
            encoder_args = {
                "encoder_model": "multi_shank_v1",
                "n_shanks": n_shanks,
                "shank_encoder_params": encoder_args,
            }
        else:
            raise ValueError(
                "shank_ids requires a simple, convolutional1D, or multi_shank_v1 "
                "encoder."
            )

    elif encoder_args["encoder_model"] in {"multi_shank_v0", "multi_shank_v1"}:
        logger.info(
            "single shank data for multi-shank encoder detected; "
            "using shank-specific encoder."
        )
        encoder_args = encoder_args["shank_encoder_params"]


    # SimpleEncoder and convolutionalEncoder1D append a tuple output layer.
    if "widths" in encoder_args:
        encoder_args["widths"] = tuple(encoder_args["widths"])
    if "shank_encoder_params" in encoder_args:
        shank_encoder = encoder_args["shank_encoder_params"]
        if shank_encoder["encoder_model"] not in {"simple", "convolutional1D"}:
            raise ValueError("Each shank must use a continuous waveform encoder.")
        shank_encoder["widths"] = tuple(shank_encoder["widths"])

    resolved.update(
        encoder_args=encoder_args,
        latent_dim=dimensions,
        context_dim=dimensions,
        n_neg_samples=n_neg_samples,
        predicted_sequence_length=1,
        return_embeddings_in_call=True,
    )
    if resolved.get("context_convolutional"):
        raise ValueError("context_convolutional is not supported by embed_data.")
    return resolved


def train_from_waveform_data(
    mark_times: np.ndarray,
    mark_waveforms: np.ndarray,
    dimensions: int,
    output_dir: Path | str,
    sample_length: int = 2000,
    model_args: dict[str, Any] | Path | str = "clusterless_multishank_default_v0",
    train_args: dict[str, Any] | None = None,
    time_scale_unit: str = "ms",
    *,
    shank_ids: np.ndarray | None = None,
    chunk_size: int = 5000,
    chunk_padding: int = 2000,
    interpolation_step: float | None = 0.001,
) -> C3poAnalysis:
    """Train, embed, and save a clusterless C3PO model from waveform features.

    This follows ``utils.dandi.train_from_spike_data``, replacing sorted unit
    indices with continuous marks. It uses the same context architecture and
    training defaults, with an MLP waveform encoder by default.

    Parameters
    ----------
    mark_times : numpy.ndarray
        Event timestamps in **seconds**, shape (n_spikes,). Events and their
        features are stably sorted by time. For repeated timestamps, only the
        first event is retained because the model requires positive delays.
    mark_waveforms : numpy.ndarray
        Finite, precomputed waveform features, shape (n_spikes, n_features).
        Examples include channel amplitudes or waveform principal components.
        Values are converted to float32 without normalization. Full waveform
        snippets must first be converted to feature vectors. Do not include a
        shank ID column; supply it through ``shank_ids`` instead.
    dimensions : int
        Positive latent and context dimensionality.
    output_dir : pathlib.Path or str
        Directory for ``args.json``, ``params.hdf5``, ``embedding.npz``,
        ``training_loss.npy``, ``context_pca_plot.png``, and ``metrics.yaml``.
        An existing saved model is never overwritten.
    sample_length : int, default=2000
        Events per training window; must be at least two. Windows use the
        50 percent overlap implemented by ``prep_training_data``.
    model_args : dict, pathlib.Path, str, or None, default=None
        Complete model configuration, registered configuration name, or JSON
        path. Default is ``clusterless_multishank_default_v0``. Both dimensions
        and the initial negative sample count are filled in for this run. The
        input configuration is copied.
    train_args : dict or None, default=None
        Overrides for ``DEFAULT_TRAIN_ARGS``, forwarded to ``train_model``.
        Batch sizes are capped at the available number of training windows.
        Only parameters and loss are retained if extra return flags are set.
    time_scale_unit : {"ms", "us", "s"}, default="ms"
        Units of the inter-event delays supplied to the model. Input event
        timestamps and all saved analysis times are always in seconds.
    shank_ids : numpy.ndarray or None, default=None
        Nonnegative integer tetrode/shank IDs, shape (n_spikes,). IDs must be
        contiguous starting at zero. If supplied, each shank gets a separate
        waveform encoder through ``multi_shank_v1``. Feature columns must have
        the same meaning within each shank. IDs are appended internally and
        are not normalized.
    chunk_size : int, default=5000
        Events per embedding chunk.
    chunk_padding : int, default=2000
        History events used to warm up each embedding chunk. Set this to cover
        the context model's history. The first ``chunk_padding`` embedded rows
        are NaN. At least ``dimensions`` finite context rows must remain.
    interpolation_step : float or None, default=0.001
        Context interpolation spacing in seconds. None skips interpolation.
        Interpolated context and PCA are available on the returned analysis;
        ``save_model`` saves event embeddings, parameters, and configuration.

    Returns
    -------
    C3poAnalysis
        Trained analysis with event times ``t`` of shape (n_events,), mark
        embeddings ``z`` and contexts ``c`` of shape (n_events, dimensions),
        and fitted context PCA. Here n_events is the number of distinct
        timestamps minus one: the first event provides the delay reference.

    Raises
    ------
    ValueError
        If shapes, values, configuration, or the available event count are
        incompatible with training and context PCA.
    FileExistsError
        If ``output_dir`` already contains a saved model.

    Notes
    -----
    Supply one continuous recording. Training windows and interpolation do
    not detect recording gaps. Distinct events with identical timestamps are
    dropped using the same positive-delay restriction as the sorted pipeline;
    their count is recorded in ``metrics.yaml``.

    Examples
    --------
    >>> from c3po.utils.clusterless import train_from_waveform_data
    >>> analysis = train_from_waveform_data(  # doctest: +SKIP
    ...     mark_times=spike_times_seconds,
    ...     mark_waveforms=normalized_channel_amplitudes,
    ...     dimensions=16,
    ...     output_dir="clusterless_model",
    ...     shank_ids=tetrode_ids,
    ...     train_args={"learning_rate": 1e-5},
    ... )
    """
    for name, value, minimum in (
        ("dimensions", dimensions, 1),
        ("sample_length", sample_length, 2),
        ("chunk_size", chunk_size, 1_000),
        ("chunk_padding", chunk_padding, 0),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    time_scales = {"s": 1.0, "ms": 1e3, "us": 1e6}
    if time_scale_unit not in time_scales:
        raise ValueError("time_scale_unit must be 'ms', 'us', or 's'.")
    if interpolation_step is not None and (
        not np.isfinite(interpolation_step) or interpolation_step <= 0
    ):
        raise ValueError("interpolation_step must be positive and finite, or None.")

    times = np.asarray(mark_times, dtype=np.float64)
    waveforms = np.asarray(mark_waveforms, dtype=np.float32)
    if times.ndim != 1 or times.size < 2 or not np.isfinite(times).all():
        raise ValueError(
            "mark_times must be a finite 1D array with at least two events."
        )
    if (
        waveforms.ndim != 2
        or waveforms.shape[0] != times.size
        or waveforms.shape[1] == 0
        or not np.isfinite(waveforms).all()
    ):
        raise ValueError(
            "mark_waveforms must be finite with shape (n_spikes, n_features)."
        )

    n_shanks = None
    if shank_ids is not None:
        ids = np.asarray(shank_ids)
        if (
            ids.shape != times.shape
            or not np.issubdtype(ids.dtype, np.number)
            or np.iscomplexobj(ids)
            or not np.isfinite(ids).all()
            or np.any(ids < 0)
            or np.any(ids != np.floor(ids))
        ):
            raise ValueError(
                "shank_ids must be nonnegative integers, shape (n_spikes,)."
            )
        unique_ids = np.unique(ids)
        if not np.array_equal(unique_ids, np.arange(unique_ids.size)):
            raise ValueError("shank_ids must be contiguous starting at zero.")
        n_shanks = int(unique_ids.size)
        waveforms = np.column_stack((waveforms, ids.astype(np.float32)))

    order = np.argsort(times, kind="stable")
    times = times[order]
    waveforms = waveforms[order]
    distinct = np.concatenate(([True], np.diff(times) > 0))
    n_duplicates = int((~distinct).sum())
    times = times[distinct]
    waveforms = waveforms[distinct]
    n_events = times.size - 1
    if n_events < sample_length:
        raise ValueError("Not enough distinct events for one full training window.")
    if n_events - chunk_padding < dimensions:
        raise ValueError(
            "Not enough events remain after chunk_padding for context PCA."
        )

    compiled_train_args = deepcopy(DEFAULT_TRAIN_ARGS)
    if train_args is not None:
        compiled_train_args.update(train_args)
    for name in ("initial_batch_size", "min_batch_size", "initial_n_neg"):
        value = compiled_train_args[name]
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError(f"train_args['{name}'] must be a positive integer.")
    resolved_args = _resolve_clusterless_model_args(
        model_args, dimensions, compiled_train_args["initial_n_neg"], n_shanks
    )

    # Keep float64 timestamps for output and float32 marks/delays for JAX.
    delays_seconds = np.diff(times)
    delta_t = (delays_seconds * time_scales[time_scale_unit]).astype(np.float32)[
        None, :
    ]
    if not np.isfinite(delta_t).all() or np.any(delta_t <= 0):
        raise ValueError(
            "Inter-event delays must remain positive and finite in float32."
        )
    x = waveforms[None, 1:, :]
    if n_events == sample_length:
        # prep_training_data uses a strict '<' bound and omits this exact fit.
        x_train, delta_t_train = x.copy(), delta_t.copy()
    else:
        x_train, delta_t_train = prep_training_data(x, delta_t, sample_length)
    compiled_train_args["initial_batch_size"] = min(
        compiled_train_args["initial_batch_size"], x_train.shape[0]
    )
    compiled_train_args["min_batch_size"] = min(
        compiled_train_args["min_batch_size"], compiled_train_args["initial_batch_size"]
    )

    output_dir = Path(output_dir)
    if any(
        (output_dir / name).exists()
        for name in ("embedding.npz", "params.hdf5", "args.json")
    ):
        raise FileExistsError(f"Model directory {output_dir} already contains a model.")
    output_dir.mkdir(parents=True, exist_ok=True)

    mua_rate = float(1.0 / np.mean(delays_seconds))
    print(f"Average multi-unit activity rate: {mua_rate:.2f} Hz")
    print(f"Average sample duration: {sample_length / mua_rate:.2f} seconds")
    if n_duplicates:
        print(f"Dropped {n_duplicates} events with repeated timestamps.")
    model = C3PO(**resolved_args)
    rand_key = jax.random.PRNGKey(0)
    params = model.init(rand_key, x_train[:10], delta_t_train[:10], rand_key)
    training_start = perf_counter()
    training_result = train_model(
        model, params, x_train, delta_t_train, **compiled_train_args
    )
    params, tracked_loss = training_result[:2]
    # Synchronize device work before recording elapsed training time.
    jax.block_until_ready(params)
    training_time = perf_counter() - training_start
    print(f"Training completed in {training_time:.2f} seconds.")

    analysis = C3poAnalysis(model=model, model_args=resolved_args, params=params)
    analysis.embed_data(
        x,
        delta_t,
        chunk_size=chunk_size,
        chunk_padding=chunk_padding,
        delta_t_units="ms" if time_scale_unit == "ms" else "s",
    )
    # Use the original seconds timestamps for every unit, including 'us',
    # which embed_data does not support when reconstructing its time axis.
    analysis.t = times[1:].copy()
    analysis.fit_context_pca()
    if interpolation_step is not None:
        analysis.interpolate_context(
            np.arange(analysis.t[0], analysis.t[-1], interpolation_step)
        )
        analysis.embed_context_pca()
    analysis.save_model(output_dir)
    np.save(output_dir / "training_loss.npy", np.asarray(tracked_loss, dtype=float))

    start = 10000 if n_events > 10000 else chunk_padding
    ind_plot = slice(start, start + 5000)
    fig, ax = plt.subplots(figsize=(12, 6))
    try:
        ax.plot(analysis.t[ind_plot], analysis.c_pca[ind_plot, :3])
        ax.set(xlabel="Time (s)", ylabel="Context PC", title="Context PCA over time")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(output_dir / "context_pca_plot.png", dpi=300)
    finally:
        plt.close(fig)

    metrics = {
        "training_time": float(training_time),
        "n_input_spikes": int(np.asarray(mark_times).size),
        "n_events": int(n_events),
        "n_duplicate_timestamps": n_duplicates,
        "n_features": int(np.asarray(mark_waveforms).shape[1]),
        "n_shanks": n_shanks,
        "dimensions": int(dimensions),
        "mua_rate": mua_rate,
        "time_scale_unit": time_scale_unit,
        "timestamp_unit": "s",
        "n_training_windows": int(x_train.shape[0]),
        "sample_length": int(sample_length),
        "initial_batch_size": int(compiled_train_args["initial_batch_size"]),
        "chunk_padding": int(chunk_padding),
        "interpolation_step": (
            None if interpolation_step is None else float(interpolation_step)
        ),
    }
    with (output_dir / "metrics.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(metrics, file, sort_keys=False)
    return analysis
