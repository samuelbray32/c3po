# Contrastive Prediction of Point Process Observations (C3PO)

Unsupervised model to infer latent generative processes from point processes occuring
in $\mathbb{R}^d$ spcae. Developed with focus for inferring neural states from waveform
features of unclustered spike events.

Analysis associated with the publication "Learning predictive latent structure from
unclustered neural spike trains" available at the
[publication repository](https://github.com/samuelbray32/c3po_publication).

## Installation

### Local

```bash
conda env create c3po
conda activate c3po
conda install pip
pip install c3po-neuro
```

Optional dependencies:

```bash
c3po-neuro[gpu] #For gpu acceleration.
c3po-neuro[cloud] #For loading of cloud models stored on higging face
```

### Try it out on the cloud

Run C3PO on public DANDI recordings in Google Colab with no local installation required. Enter the Dandiset ID and the path to an NWB file containing sorted spikes. The notebook sets up C3PO in the Colab runtime, trains a model on a GPU, and displays a PCA plot of the inferred context. You can download the trained model and analysis outputs as a ZIP archive or save them to Google Drive. [Open the C3PO demo in Google Colab](https://colab.research.google.com/github/samuelbray32/c3po/blob/main/tutorial/c3po_dandi_colab.ipynb).

## Quick Start

### Sorted Data [(Tutorial)](./tutorial/sorted_units.ipynb)

C3PO provides functions for constructing and training sorted spike models from several sources:

- Data published on the DANDI archive:  `c3po.utils.train_from_dandi()`
- Data stored within a local NWB file: `c3po.utils.train_from_nwb()`
- A loaded array of spike times and neuron ids: `c3po.utils.train_from_sorted_data()`

By default, these will train with the architecture and parameters used in the publication data screening. Alternative model architectures can be passed as arguments. For a walk through of these options, see the [deep-dive tutorial notebook](./tutorial/sorted_units_deep_dive.ipynb)

### Clusterless Data [(Tutorial)](./tutorial/clusterless.ipynb)

C3PO can infer latent neural states directly from spike times and waveform features, without spike sorting. Each event is represented by its timestamp and a feature vector, such as peak amplitudes across recording channels.

From extracted waveform features, use the function

- `c3po.utils.clusterless.train_from_waveform_data()`

Alternative model architectures can be passed as arguments. For a walk through of some of these options, see the [deep-dive tutorial notebook](./tutorial/sorted_units_deep_dive.ipynb)

## Details

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
    rate_model="sharedSpace",
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

**NCE (DEFAULT)**: The noise contrastive estimation loss is given by:
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
loss =  model.contrastive_loss(
    pos_params,
    neg_params,
    delta_t,
    z[:, 1:],
    neg_z,
)
```

**MLE**: Maximum likelihood loss is given by
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
loss = model.mle_loss(
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
