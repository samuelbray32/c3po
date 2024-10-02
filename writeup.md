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

### Contrastive Coding
