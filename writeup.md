# C3PO

## Motivation

- Studying the structure of population neural dynamics can help identify brain states and reveal how the brain encodes information about the world.
- "Population neural activity" is a very-high dimensional vector, typically thought of as the firing rate of each neuron.
  - Due to connectivity, learning, etc., the actual firing rate vectors don't fully span the high-dimensional space but occur on some lower-dimensional manifold.
  - Interpreting this data involves embedding the manifold of observed population activity into a lower-dimensional space.
- Most existing methods of embedding the dynamics (e.g., CEBRA, UMAP) take sorted spike firing rate vectors as input.
  - This requires spike-sorting the electrical series data, which can be time-intensive and exclude low-amplitude events.
- C3PO provides a 'clusterless' method to identify latent neural states from population activity.

## Model

### Observations

The first step of the pipeline is to extract waveform mark events from electrophysiology data. This can be done with threshold detection to identify event times ($t$), followed by "waveform" featurization of each event ($W$). For example, this could be the maximum voltage amplitude on each electrode channel within 50 ns of the detected event.

This gives us the set of observations $X = \{(t_i,W_i)\}_{i=1}^{n_{obs}}$. Since these are ordered observations, this can also be described as $X = \{(\Delta t_i,W_i)\}_{i=1}^{n_{obs}}$ where $\Delta t_i = t_i - t_{i-1}$, or the wait time between subsequent events.

With increasing use of high-density electrode probes for electrophysiology, the waveform feature $W \in \mathbb{R}^{D_{obs}}$ can exist in a very high-dimensional space, creating issues for clusterless algorithms which depend on distance metrics.

### Embedding

This motivates us to find a lower-dimensional embedding of the waveforms $Z_i = \epsilon(W_i)$ where $Z_i \in \mathbb{R}^{D_{embed}}$ and $D_{embed} < D_{obs}$. In order to study latent states of the underlying neural system, we want this embedding to preserve functional information encoded by the neuron's firing. We will discuss the appropriate loss to achieve this below.

The implemented loss function does not ultimately depend on the functional form of $\epsilon(W_i)$ provided that a gradient can be back-propagated through its transformation. Therefore, we can conceive of it as a multilayer dense neural network, though this does not preclude implementation of other forms which take more explicit features of the probe device (TODO: implement geometry dependence).

### Context

The embedded waveform feature $Z$ then contains information about an individual clusterless mark. However, individual neuron spiking is a noisy, stochastic process. Moreover, we are often interested in identifying neural states that vary on much slower timescales than an individual firing event and are defined by collective activity of multiple neurons (e.g., hippocampal representation of position, value coding in a multi-choice task). We therefore want to identify a context feature $C_i$ which encodes information about collective neural activity over time.

We can functionally define $C_i = \rho(\Delta t_i, Z_i, C_{i-1})$, where $\rho$ is a recurrent neural network that takes $(\Delta t_i, Z_i)$ as input at each timepoint and carries the history term $C_i$. We include $\Delta t_i$ in the input to the neural network as both the firing neuron's identity (encoded in $Z_i$) and its firing rate (approximately encoded by $\Delta t_i$) are determined by the latent context of the network. Again, the loss term we describe below is not dependent on the exact form of this transformation, allowing us to use variations of recurrent neural networks such as LSTM in our implementation.

### Rate Prediction

Similar to [Contrastive Predictive Coding (CPC)](https://arxiv.org/abs/1807.03748), our definition of a "good" context embedding is based on its ability to predictively identify the future state of the system. Explicitly, we need to define a function $r(\theta(Z_i,C_{i-1})) \sim P(Z_i|C_{i-1})$ which is proportional to the likelihood of an embedded observation $Z_i$ in a given context $C$.

Again, the final loss term is independent of the functional form of the parameterization $\theta(Z_i,C_{i-1})$. Matching the approach of the original CPC paper, our default implementation is a bilinear model for each needed parameter $\theta_i = Z_i B C_{i-1}$ where $B$ is a learnable parameter matrix, though other implementations are provided.

### Loss

To summarize the previous architecture, we have embedded our sequence of waveform observations $X = \{(\Delta t_i,W_i)\}_{i=1}^{n_{obs}}$ independently into a sequence of events $Z$ in a lower-dimensional space. This sequence is then iterated over by an RNN to generate a series of context states $C$.

#### Defining the probability model

We can now define the likelihood of our observations. Qualitatively, we can define the probability of each observation as a spike with the given waveform $W_i$ with a wait time of $\Delta t_i$, with no other spike events during that wait time. This can be written as $P(X_i) = P(W_i,\omega=\Delta t_i) \cdot \prod_{j \neq i} P(W_j, \omega > \Delta t_i)$ where $\omega$ is the wait time for an event.

# Notes on training hyperparameters

- `n_neg_samples`:
  - _Low values_: less specific prediction required. Need to know when rates are high, but less sensitive to false positives.
  - _High values_: requires more precision when predicting when a unit fires. Loss term is much more punished for predicting high rates at inappropriate times.
- `batch size`:
  - Changes what your loss is contrasting against:
  - _High values_: Requires that spikes from a trial are different from different states and different trials. Less pressure for contrast within-trial.
  - _Low values_: More of the contrastive loss comes from within-trial spikes. Model has to learn more about differences over time in a trial.

- Recommended training protocol:
  - Annealing of `n_neg_samples`:
    - Start with a low value (e.g., 8) to allow the model to learn general rates and embeddings. Allow to train until improvement <1%.
    - Double `n_neg_samples` and train until stable.
    - Repeat up to max value (128).
  - Preliminary: `batch_size`:
    - Run protocol above once with `batch_size`=64 to learn general structure.
    - Repeat protocol with `batch_size`=8 for refinement of within-trial changes.
    - TODO: verify whether the initial stage is necessary.
