import numpy as np
try:
    from spyglass.common import FirFilterParameters
except ImportError:
    print("Spyglass not found. Please install spyglass to use the band filter functionality.")
import pandas as pd
from scipy.signal import hilbert

def filter_data(
    t_data: np.ndarray,
    data: np.ndarray,
    filter_coeff: np.ndarray,
    time_windows: np.ndarray,
    context_dim: int,
) -> dict:
    """Band-pass filter data and compute its analytic phase and power.

    Parameters
    ----------
    t_data : np.ndarray
        Timestamps of `data`, shape (n_samples,).
    data : np.ndarray
        Data to filter, shape (n_samples, n_dims).
    filter_coeff : np.ndarray
        FIR filter coefficients (as used by spyglass FirFilterParameters).
    time_windows : np.ndarray
        Intervals (n_intervals, 2) over which to filter.
    context_dim : int
        Number of context dimensions. Currently unused.

    Returns
    -------
    dict
        "signal" : filtered data, "time" : timestamps, "phase" : Hilbert phase
        in [0, 2*pi], "power" : squared Hilbert amplitude. Samples with NaNs
        are removed; arrays have shape (n_valid_samples, n_dims).
    """
    f, t_f = FirFilterParameters().filter_data(
        t_data,
        data,
        filter_coeff,
        time_windows,
        np.arange(data.shape[1]),
        decimation = 1,
    )
    ind = ~np.isnan(f).any(axis=1)
    f = f[ind]
    t_f = t_f[ind]

    analytic_signal_df = pd.DataFrame(
        hilbert(f, axis=0),
        index=pd.Index(t_f, name="time"),
    )
    phase_df = pd.DataFrame(
        np.angle(analytic_signal_df) + np.pi,
        columns=analytic_signal_df.columns,
        index=analytic_signal_df.index,
    )
    power_df = pd.DataFrame(
        np.abs(analytic_signal_df) ** 2,
        columns=analytic_signal_df.columns,
        index=analytic_signal_df.index,
    )

    return dict(
        signal=f,
        time=t_f,
        phase=phase_df.values,
        power=power_df.values,
    )

