import numpy as np

from .phase import generate_waveform_features


def generate_step_periodic_spike_train(
    latent_period: float = 3,
    noise_scale: float = 0.1,
    t_max: float = 10000,
    n_units: int = 2,
    n_channels: int = 32,
    max_wait_update: float = None,
):

    rate_scale = 10
    template_waveforms = generate_waveform_features(n_channels, n_units)

    # generate spike train
    t0 = 0
    mark_times = []
    mark_ids = []
    marks = []
    rates = np.zeros(n_units)

    if max_wait_update is None:
        max_wait_update = latent_period / 10
    while t0 < t_max:
        # get wait time
        g = np.sin(t0 / latent_period * (2 * np.pi))
        rates[: n_units // 2] = rate_scale * (g < 0).astype(float)
        rates[n_units // 2 :] = rate_scale * (g > 0).astype(float)

        cum_rate = np.sum(rates)
        wait_time = np.random.exponential(1 / cum_rate)

        # resample with updated rates if wait time is too long
        if wait_time > max_wait_update:
            t0 += max_wait_update
            print("waited")
            continue

        # get event id
        t0 += wait_time
        rates.shape
        p_unit = rates / rates.sum()
        if (total_p := p_unit[:-1].sum()) > 1:
            p_unit = p_unit * (1 / (total_p + 1e-8))
        p_unit[-1] = 1 - p_unit[:-1].sum()
        unit_id = np.random.choice(rates.shape[0], p=p_unit)

        # store values
        mark_times.append(t0)
        mark_ids.append(unit_id)
        marks.append(
            template_waveforms[unit_id] + np.random.normal(0, noise_scale, n_channels)
        )

    mark_ids = np.array(mark_ids)
    mark_times = np.array(mark_times)
    marks = np.array(marks)

    return mark_ids, mark_times, marks, template_waveforms
