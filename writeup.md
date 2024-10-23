# C3PO

## Motivation

- Studying the structure of population neural dynamics can help identify brain states and reveal how the brain encodes information about the world
- "Population neural activity" is a very-high dimensional vector, typically thought of as the firing rate of each neuron
  - Due to connectivity/ learning/ etc. the actual firing rate vectors don't fully span the high dimensional space but occur on some lower-dimensional manifold
  - Interpreting this data involves embedding the manifold of observed population activity into a lower-dimensional space
- Most existing methods of embedding the dynamics (e.g. CEBRA, UMAP) take sorted spike firing rate vectors as input
  - This requires spike-sorting the electrical series data, which can be time-intensive and exclude low-amplitude events
- C3PO provides a 'clusterless' method to identify latent neural states from population activity

## Model

### Observations

The first step of the pipeline is to extract waveform mark events from electrophysiology data. This can be down with threshold detection to identify event times ($t$), followed by "waveform" featurization of each event ($W$).  For example, this could be the maximum voltage amplitude on each electrode channel within 50ms of the detected event.

This gives us the set of observations $X = \{(t_i,W_i)\}|_{i=1}^{n_{obs}}$. Since these are ordered observations, this can also be described as $X = \{(\Delta t_i,W_i)\}|_{i=1}^{n_{obs}}$ where $\Delta t_i = t_i - t_{i-1}$, or the wait time between subsequent events.

With increasing use of high-density electrode probes for electrophysiology, the waveform feature $W \in {\rm I\!R}^{D_{obs}}$ can exist in a very high-dimensional space, creating issues for clusterless algorithms which depend on distance metrics.

### Embedding

  This motivates us to find a lower-dimensional embedding of the waveforms $Z_i = \Epsilon(W_i)$ where $Z_i \in {\rm I\!R}^{D_{embed}}$ and $d_{embed} < d_{obs}$. In order to study latent states of the underlying neural system, we want this embedding to preserve functional information encoded by the neuron's firing. We will discuss the appropriate loss to achieve this below

  The implemented loss function does not ultimately depend on the functional form of $\Epsilon(W_i)$ provided that a gradient can be back-propagated through its transformation. Therefore, we can conceive of it as a multilayer dense neural network, though this does not preclude implementation of other forms which take more explicit features of the probe device (TODO: implement geometry dependence).

### Context

  The embedded waveform feature $Z$ then contains information about an individual clusterless mark. However individual neuron spiking is a noisy, stochastic* process. Moreover, we are often interested in identifying neural states that vary on much slower timescales than an indivdual firing event and are defined by collective activity of multiple neurons (e.g. hippocampal representation of position, value coding in a multi-choice task). We therefore want to identify a context feature $C_i$ which encodes information about collective neural activity over time.

  We can functionally define $C_i = \rho(\Delta t_i, Z_i, C_{i-1})$. Where $\rho$ is a recurrent neural network that takes $(\Delta t_i, Z_i)$ as input at each timepoint and carries the history term $C_i$. We include $\Delta t_i$ in the input to the neural network as both which neuron that fires (encoded in $Z_i$) and it's firing rate (~ encoded by $\Delta t_i$) are determined by the latent context of the network.  Again, the loss term we describe below is not dependent on the exact form of this transformation, allowing us to use variations of recurrent neural networks such as LSTM in our implementation.
