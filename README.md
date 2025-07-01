# Contrastive Prediction of Point Process Observations (C3PO)

Unsupervised model to infer latent generative processes from point processes occuring
in $\mathbb{R}^d$ spcae. Developed with focus for inferring neural states from waveform
features of unclustered spike events.

## Usage

### Model declaration

In both intuition and implementation, the c3po model is broken into three modules.
Details and derivation of the overall architecture and role of each of these components
is described in the [documentation](docs/writeup.md):

- [encoder model](src/c3po/model/encoder.py): Accepts the waveform feature vector $W$
and embeds it into a lower `latent_dim` space as the embedded mark vector $Z$.
- [context model](src/c3p0/model/context.py): Embeds the history of marks
$\set{Z_j}_{j=1}^i$ into a context feature vector $C_i$
- [rate-prediction model](src/c3po/model/rate_prediction.py): Estimates the set of
context-dependent parameters $\theta(Z_i,C_j)$ used to calculate the hazard function of
a given embedded waveform $H(\theta(Z_j, C_i),\Delta t_i)$. Implemented options for the
emission process model are found [here](src/c3po/model/process_models.py).

The complete model is agnostic to the deep learning architecture used to implement each
of these modules. For example, the context model can be implemented as any history-
coding model including a RNN, a wavenet architecture, or causal transformers. To
specify which of the the implemented versions of each module to use, each has a factory
constructor which accepts a key for the architecture type and a dictionary or arguments
for the module (e.g. number of layers, convolutional filter size, etc.).

A complete c3po model for training is defined by the [C3PO class](src/s3po/model/model).
The initialization of the class requires constructor arguments for each of the modules
and definition of the embedded dimension size. An example is given below.

```python
# hyperparams
latent_dim = 10
context_dim = 10
# encoder
encoder_widths = [128, 128, 64]
encoder_args = dict(
    encoder_model="simple",
    widths=encoder_widths,
)
# context
dilations = [1, 2, 4, 8, 16]
kernels = [10, 20, 64, 64, 128]
dilations = dilations * 2
kernels = kernels * 2
context_args = dict(
    context_model="wavenet",
    layer_dilations=dilations,
    layer_kernel_size=kernels,
    expanded_dim=32,
    smoothing=10,
)
# rate model
rate_args = dict(
    rate_model="bilinear",
)
distribution = "poisson"
n_neg_samples = 128
model = C3PO(
    encoder_args,
    context_args,
    rate_args,
    distribution,
    latent_dim,
    context_dim,
    n_neg_samples,
)
```

### Loss functions

*Note:* implementation is correct and functional but require code cleanupof naming and
documentation. For derivation and explanation of loss functions see
[documentation](/docs/contrastive_loss.md)

__NCE (Recommended)__: The noise contrastive estimation loss is given by:
$\mathcal{L} = - \mathbb{E}_i[\log \frac{H_i/H'_i}{\sum_j H_j/H'_j}]$. Example execution
of the `C3PO` method shown below:

```python
#define model
model = C3PO(
    encoder_args,
    context_args,
    rate_args,
    distribution,
    latent_dim,
    context_dim,
    n_neg_samples,
    return_embeddings_in_call=True,
)
run_model = jax.jit(model.apply)

# forward pass to get \theta params and embedded values
pos_params, neg_params, z, c, neg_z = run_model(params, x, delta_t, rand_key)
# loss
loss =  model.contrastive_sequence_loss(
    pos_params,
    neg_params,
    delta_t,
    z[:, 1:],
    neg_z,
)
```

__MLE__: Maximum likelihood loss is given by
$\mathcal{L}=\mathbb{E}_i[-\log H_i+\log\bar{S}_i]$ and is provided as a method of the
`C3PO` class. Example execution shown below:

```python
# define model
model = C3PO(
    encoder_args,
    context_args,
    rate_args,
    distribution,
    latent_dim,
    context_dim,
    n_neg_samples,
    predicted_sequence_length,
)
run_model = jax.jit(model.apply)
# forward pass to get \theta params
pos_params, neg_params = run_model(params, x, delta_t, rand_key)
# loss
loss = model.loss_generalized_model(
    pos_params, neg_params, delta_t,
)
```

## Similar Methods

- Contrastive Predictive Coding (CPC)
  - Works for continuous time rather than point processes
- Temporal Neural Networks (e.g. Hawkes process)
  - designed for categorically labeled events
- CEBRA
  - designed for sorted spiking data
  - semi-supervised definition of matching states
