# Model Architecture

## Recovering latent structure from marked point-process observations (Fig. 1A)

We now construct a model that inverts the generative process described above. In the generative view, latent neural dynamics $G_t$ give rise to firing rates $\lambda(G_t)$, from which spikes are stochastically generated and subsequently observed as waveform-marked events. Our goal is to learn a representation that recovers the predictive structure of $G_t$ directly from these observations.

Concretely, given a sequence of marked point-process observations

$$
X = \{(\Delta t_i, W_i)\},
$$

we seek to learn a transformation of the form

$$
(\Delta t_i, W_i) \;\to\; Z_i \;\to\; C_i,
$$

where $Z_i$ represents a low-dimensional embedding of individual events, and $C_i$ represents a latent context variable that approximates the predictive state of the system. In this framework, $C_i$ plays the role of an inferred sufficient statistic of the history, analogous to the latent variable $G_t$ in the generative model.

---

## Mark embedding

The first stage of the model constructs a low-dimensional embedding of the observed waveform features. Each waveform $W_i \in \mathbb{R}^{D_{\text{obs}}}$ is mapped to a latent representation:

$$
Z_i = \epsilon(W_i), \quad Z_i \in \mathbb{R}^{D_{\text{embed}}}, \quad D_{\text{embed}} < D_{\text{obs}}.
$$

This embedding serves as a representation of the underlying event identity and its functional role within the neural population.
Importantly, unlike traditional spike sorting approaches, this representation is not constrained to identify discrete neuron identities nor encode waveform similarity; rather, it is learned to support prediction of future events, and thus organizes observations according to their firing timing

The transformation $\epsilon(\cdot)$ may be implemented using a flexible neural network architecture (e.g., multilayer perceptron or convolutional encoder), provided that gradients can be propagated through the mapping. This flexibility allows the model to accommodate diverse electrode geometries and recording configurations.

From the perspective of the generative model, this stage corresponds to inferring a latent representation of the emission process underlying each observed waveform.

---

# Context Representation of Latent Dynamics

To capture the temporal structure of neural activity, we construct a context variable $C_i$ that integrates information across past events:

$$
C_i = \rho(\{\Delta t_j, Z_j\}_{j \le i}),
$$

where $\rho$ is a sequence model that maps the history of observed events to a fixed-dimensional representation.

This formulation generalizes the recursive update

$$
C_i = \rho(\Delta t_i, Z_i, C_{i-1}),
$$

and allows the context to be implemented using a wide class of architectures. In general, $\rho$ may be any function that takes a sequence of spike events as input and produces a vector-valued representation, including recurrent neural networks, temporal convolutional networks, or transformer-based models.

In this work, we use a WaveNet-style architecture, consisting of stacks of dilated causal convolutions. This architecture is particularly well-suited for neural spiking data, as it efficiently captures long-range temporal dependencies while maintaining causal structure. By exponentially expanding the receptive field with depth, dilated convolutions enable the model to integrate information over extended time horizons without the computational cost associated with recurrent or attention-based models.

The resulting context variable $C_i$ represents a learned summary of population activity that evolves over time. By construction, it is optimized to retain information that is predictive of future events, and can therefore be interpreted as an estimate of the latent state $G_t$, up to transformations that preserve predictive equivalence.

---

## Conditional hazard and rate modeling

In the generative process, the latent state $G_t$ determines a set of continuously varying firing rates $\lambda(G_t)$, which define the stochastic generation of spike events. In contrast, our observations are discrete events drawn from a point process. Modeling their likelihood therefore requires specifying a conditional hazard function:

$$
H(X_i \mid C_{i-1}),
$$

which specifies the probability of observing an event $X_i$ given the preceding context.

We parameterize this hazard function through a scoring function $\theta(Z_i, C_{i-1})$, computed using a bilinear interaction between the mark embedding and the context:

$$
\theta(Z_i, C_{i-1}) = Z_i^\top R C_{i-1}.
$$

This score is then mapped to a valid hazard function according to a chosen point-process model. In general, any parametric form with a closed-form hazard function can be used within this framework. For examples presented in this work, we use a Poisson process model, where the hazard is equal to the conditional firing rate:

$$
H(X_i \mid C_{i-1}) = \hat{\lambda}(X_i \mid C_{i-1}) = Z_i^\top R C_{i-1}.
$$

For simplicity and interpretability, we further constrain the model such that the embedding and context spaces have equal dimensionality, and set $R = I$, yielding:

$$
\hat{\lambda}(X_i \mid C_{i-1}) = Z_i^\top C_{i-1}.
$$

This shared-space formulation places both marks and contexts in a common latent space, where their inner product directly determines event likelihood. This structure will later enable a geometric interpretation of the learned representations.

---

## Relationship to the generative model

The model architecture provides a structured approximation to the generative process described in Fig. 1A. The latent state $G_t$ is replaced by the learned context representation $C_t$, the emission process is captured by the mark embedding $Z$, and the firing rate field $\lambda(G)$ is replaced by the parameterized hazard function $H(X \mid C)$.

$$
G_t \;\leftrightarrow\; C_t, \quad
\lambda(Z \mid G) \;\leftrightarrow\; H(X \mid C), \quad
\text{emission identity} \;\leftrightarrow\; Z.
$$

Rather than explicitly recovering $G_t$ the model learns representations $C_t$ and $Z_i$ that are sufficient for predicting future observations, thereby capturing the predictive structure of the underlying neural dynamics.
