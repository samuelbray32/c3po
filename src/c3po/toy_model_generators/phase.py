import numpy as np
import matplotlib.pyplot as plt
import os


def firing_rate(
    theta_t: float, theta_max: float, sigma: float, amplitude: float
) -> float:
    """Compute the firing rate of a neuron at a given angle theta_t.

    Args:
        theta_t float: Angle at which to compute the firing rate.
        theta_max float: Preferred angle of the neuron.
        sigma float: Standard deviation of the tuning curve.
        amplitude float: Maximum firing rate of the neuron.

    Returns:
        float: Firing rate of the neuron at angle theta_t.
    """
    delta_theta = np.abs(np.mod(theta_t - theta_max + np.pi, 2 * np.pi) - np.pi)
    # Compute the Gaussian function value at theta_t
    f = np.exp(-0.5 * (delta_theta / sigma) ** 2) * amplitude
    return f


def make_tuning_curve_set(n_units: int = 16) -> tuple:
    """Generate a set of tuning curves for a population of neurons.

    Args:
        n_units (int, optional): number of neurons. Defaults to 16.

    Returns:
        tuple: Tuple containing the preferred angles, standard deviations and maximum firing rates of the neurons.
    """
    theta_max = np.random.uniform(0, 2 * np.pi, n_units)
    sigma = np.random.uniform(0.1, 0.5, n_units)
    amplitude = np.random.uniform(3, 10, n_units)
    return theta_max, sigma, amplitude


def get_all_rates(
    theta_t: float, theta_max: np.ndarray, sigma: np.ndarray, amplitude: np.ndarray
):
    """Compute the firing rates of a population of neurons at a given angle theta_t.

    Args:
        theta_t (float): current angle
        theta_max (np.ndarray): array of preferred angles
        sigma (np.ndarray): array of standard deviations
        amplitude (np.ndarray): array of maximum firing rates

    Returns:
        np.ndarray: array of firing rates
    """
    n_units = len(theta_max)
    f = np.zeros_like(theta_max)
    for i in range(theta_max.size):
        f[i] = firing_rate(theta_t, theta_max[i], sigma[i], amplitude=amplitude[i])
    return f


def gaussian(
    x,
    mu,
    sig,
):
    return np.exp(-np.power((x) - mu, 2.0) / (2 * np.power(sig, 2.0)))


def generate_waveform_features(n_channels: int = 32, n_units: int = 16):
    """Generate waveform features for a population of neurons.

    Args:
        n_channels (int, optional): number of probe channels. Defaults to 32.
        n_units (int, optional): number of generative neurons. Defaults to 10.

    Returns:
        np.ndarray: Array of waveform features.
    """
    peak_rng = (0.5, 1.5)  # range scale for neuron peak amplitude
    std_rng = (0.5, 3)  # range scale for gaussian spread of amplitude across channels

    template_waveforms = []
    for i in range(n_units):
        peak = np.random.uniform(*peak_rng)
        std = np.random.uniform(*std_rng)
        peak_loc = np.random.randint(0, n_channels)
        val = peak * gaussian(
            np.arange(n_channels),
            peak_loc,
            std,
        )
        template_waveforms.append(val)

    return np.array(template_waveforms)


def generate_periodic_spike_train(
    latent_period: float = 3,
    noise_scale: float = 0.1,
    t_max: float = 10000,
    n_units: int = 16,
    n_channels: int = 32,
    max_wait_update: float = None,
):
    """Generate a spike train driven by a periodic latent variable.

    Args:
        latent_period (float, optional): Period of the latent variable. Defaults to 3.
        noise_scale (float, optional): Standard deviation of the noise added to the waveforms. Defaults to .1.
        t_max (float, optional): Duration of the spike train. Defaults to 10000.
        n_units (int, optional): Number of neurons. Defaults to 16.
        n_channels (int, optional): Number of probe channels. Defaults to 32.
        max_wait_update (float, optional): Maximum wait time before updating the firing rates. Defaults to 1/10 the latent period.

    Returns:
        tuple: Tuple containing the spike times, neuron ids, waveforms, template waveforms and tuning curves.
    """

    theta_max, sigma, amplitude = make_tuning_curve_set(n_units)
    template_waveforms = generate_waveform_features(n_channels, n_units)

    # generate spike train
    t0 = 0
    mark_times = []
    mark_ids = []
    marks = []

    if max_wait_update is None:
        max_wait_update = latent_period / 10
    while t0 < t_max:
        # get wait time
        theta_t = np.mod(t0, latent_period) / latent_period * 2 * np.pi
        rates = get_all_rates(theta_t, theta_max, sigma, amplitude)
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
        # print(t0, unit_id)

    mark_ids = np.array(mark_ids)
    mark_times = np.array(mark_times)
    marks = np.array(marks)

    tuning_curves = {"theta_max": theta_max, "sigma": sigma, "amplitude": amplitude}
    return mark_ids, mark_times, marks, template_waveforms, tuning_curves
