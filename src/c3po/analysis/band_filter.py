import numpy as np
from spyglass.common import FirFilterParameters
import pandas as pd
from scipy.signal import hilbert

def filter_data(t_data, data, filter_coeff, time_windows, context_dim, n_jobs):
    f, t_f = FirFilterParameters().filter_data(
        t_data,
        data,
        filter_coeff,
        time_windows,
        np.arange(context_dim),
        n_jobs,
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

