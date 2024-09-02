import numpy as np
import jax
import jax.numpy as jnp

from functools import partial


def generate_hmm_spike_train(
    stickiness: float = 0.999,
    t=np.linspace(0, 100, 100),
    n_channels=32,
    n_units=10,
    peak_rng=[0.1, 1],
    std_rng=[0.5, 3],
    firing_rate_rng=[0, 1],
    noise_scale=0.1,
    soft_binarize_firing_rate=True,
):
    """Generate a spike train using a hidden markov model.

    Args:
        stickiness (float, optional): stickness of the markov state. Defaults to 0.999.
        t (_type_, optional): timepoints to sample the HMM. Defaults to np.linspace(0, 100, 100).
        n_channels (int, optional): number of probe channels. Defaults to 32.
        n_units (int, optional): number of generative neurons. Defaults to 10.
        peak_rng (list, optional): range scale for neuron peak amplitude. Defaults to [0.1, 1].
        std_rng (list, optional):  range scale for gaussian spread of amplitude across channels. Defaults to [0.5, 3].
        firing_rate_rng (list, optional): range of sampleable firing rates. Defaults to [0, 1].
        noise_scale (float, optional): gaussian noise added to sample mark waveforms. Defaults to 0.1.
        soft_binarize_firing_rate (bool, optional): whether to make the firing rates more binary with hmm state. Defaults to True.

    Returns:
        _type_: _description_
    """
    # generate markov sequence
    hmm = HMM(stickiness=stickiness)
    delta_t = jnp.mean(np.diff(t))
    x0 = jnp.array([1, 0])
    rand_list = jax.random.split(jax.random.PRNGKey(0), len(t))
    hmm_gen = jax.jit(partial(hmm.forward, delta_t=delta_t))
    final, hmm_sequence = jax.lax.scan(hmm_gen, init=x0, xs=rand_list)

    # generate waveform features
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

    firing_rates = np.random.uniform(*firing_rate_rng, (n_units, 2))
    template_waveforms = np.array(template_waveforms)
    if soft_binarize_firing_rate:
        firing_rates = np.exp(firing_rates * 100)
        firing_rates = firing_rates / np.sum(firing_rates, axis=1)[:, None]

    # generate spike train
    t0 = 0
    mark_times = []
    mark_ids = []
    marks = []

    while t0 < t.max():
        # get wait time
        ind = np.digitize(t0, t)
        state = np.array(hmm_sequence[ind])
        rates = state * firing_rates
        cum_rate = np.sum(rates)
        wait_time = np.random.exponential(1 / cum_rate)

        # get event id
        t0 += wait_time
        rates.shape
        p_unit = rates.sum(axis=1) / rates.sum()
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

    return mark_ids, mark_times, marks, template_waveforms, firing_rates, hmm_sequence


class HMM:
    def __init__(self, stickiness=0.9):
        self.transition = jnp.array(
            [[stickiness, 1 - stickiness], [1 - stickiness, stickiness]]
        )

    @partial(jax.jit, static_argnums=(0,))
    def forward(
        self,
        x,
        rand_key,
        delta_t,
    ):
        # rand_key = carry["rand_key"]
        p_state = jnp.dot(self.transition * delta_t, x)
        # new_rand_key, _ = jax.random.split(rand_key)
        state = jax.random.categorical(rand_key, p_state)
        state = jax.nn.one_hot(state, 2).astype(jnp.int32)
        return state, state


def gaussian(
    x,
    mu,
    sig,
):
    return np.exp(-np.power((x) - mu, 2.0) / (2 * np.power(sig, 2.0)))
