# A Generative Model of Neural Spike Train Observations (Fig 1A)

We begin by outlining an intuitive generative model for how neural population activity gives rise to the spike train observations analyzed in this work.

At any moment in time, the activity of a neural population reflects the accumulated history of internal dynamics, external inputs, and ongoing computation. Rather than treating this history as an unstructured sequence of past events, we assume that it can be summarized by a latent variable $G_t$, which acts as a sufficient statistic of the system history for predicting future activity. Formally, $G_t$ captures all information from the past that is relevant for determining the distribution of future spiking events. This concept is closely related to the notion of *causal states* in computational mechanics, where histories are grouped according to their predictive equivalence, and to broader results showing that minimal sufficient statistics preserve predictive information between past and future.

## Conditional Intensity Function

Within this framework, the probability that a given neuron emits a spike at time $t$ is governed by a conditional intensity (hazard) function:

$$
\lambda_n(t) = \lambda_n(G_t),
$$

where $n$ indexes neurons. This mapping can be interpreted as a firing rate field defined over the latent state $G$. In analogy to classical tuning curves, which describe firing rates as functions of experimentally controlled variables, here each neuron exhibits a (generally nonlinear) response function over the latent generative state. These functions may be highly nonlinear and reflect mixed selectivity to multiple underlying variables.

## Latent Dynamical Systems Perspective

To build intuition for the role of $G$, it is helpful to view neural population activity through the lens of low-dimensional dynamical systems. A growing body of work has shown that high-dimensional neural activity often evolves on low-dimensional manifolds governed by latent dynamics. In this perspective, $G_t$ corresponds to a point in this latent state space, whose evolution captures the computation being performed by the circuit.

For example, in hippocampus, the activity of place cells is driven by a combination of sensory inputs, memory, and movement history, yet can be well described as a function of a low-dimensional latent variable such as spatial position. In this case, $G_t$ can be interpreted as the animal’s position (or a related internal estimate), and each neuron’s firing rate field $\lambda_n(G)$ corresponds to its place field. More generally, in different brain regions, latent variables may encode mixtures of task variables, internal states, or dynamical trajectories, giving rise to distributed and context-dependent firing patterns.

## Spike Generation as a Point Process

Given these hazard functions, spikes are generated as stochastic events drawn from a point process. That is, conditioned on the latent state $G_t$, each neuron emits spikes probabilistically according to its instantaneous firing rate $\lambda_n(G_t)$, consistent with standard point-process formulations of neural activity. This stochastic sampling reflects both intrinsic neural variability and unobserved influences on the system.

## Observations as a Marked Point Process

Finally, neural recordings do not directly observe neuron identities or spike events in isolation. Instead, extracellular electrodes measure voltage deflections produced by nearby neurons, resulting in observed waveform snippets whose shapes depend on the spatial and biophysical properties of the underlying sources. Each detected event therefore consists of a time of occurrence and an associated waveform.

We represent the resulting data as a sequence of marked point-process observations:

$$
X = {(\Delta t_i, W_i)},
$$

where $\Delta t_i$ denotes the inter-event interval and $W_i$ denotes the recorded waveform features.

These observations arise from a cascade of transformations: Latent neural dynamics $G_t$ determine firing probabilities, spikes are stochastically generated from these probabilities, and electrode measurements produce high-dimensional, mixed observations of these events

## Summary and Challenge

This generative view highlights the central challenge: the latent state $G_t$ that governs neural activity is not directly observed, and must be inferred from indirect, noisy, and high-dimensional measurements. The following sections develop a framework for learning representations that recover this latent predictive structure directly from the observed spike train.
