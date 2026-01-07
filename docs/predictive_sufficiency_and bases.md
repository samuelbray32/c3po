
### Predictive sufficiency of the context variable

The contrastive loss derived above admits a natural interpretation in information-theoretic terms. Because the loss depends only on the relative hazard $(H(Z_j,\Delta t_i\mid C_{i-1}) / H'(Z_j,\Delta t_i))$, optimizing the model corresponds to learning a representation $(C_{i-1})$ that is sufficient to discriminate the true next mark from negative samples drawn from a reference distribution. At the population optimum of noise-contrastive estimation, the scoring function implicit in the hazard ratio recovers the log-density ratio between the true conditional mark distribution and the alternative model.

Formally, if the next observed mark ($Z_i$) is generated according to a latent generative state ($G_{i-1}$), such that:

$$p(Z_i \mid \text{history}) = p(Z_i \mid G_{i-1}) \tag{1}$$

then any representation ($C_{i-1}$) that minimizes the contrastive loss must retain all information from the history that is predictive of ($Z_i$). At optimum, ($C_{i-1}$) functions as a predictive sufficient statistic of the event history: conditioned on ($C_{i-1}$), no additional information from the past can further improve prediction of the next mark.

From [eq. 1], this means that the information reatined by $C$ after optimizing C3PO is _equivalent_ to that defining the generative process $G$.
This does not mean that $C$ is a 1:1 mapping of $G$. We discuss how the choice of functional forms in C3PO contrains the relationship between these values below.

---

### Linearization of prediction and basis representations over latent generative variables

The predictive sufficiency of ($C$) is tightly linked to its role as a basis representation over the latent generative space. In the model described above, the hazard function is parameterized as a function of the inner product between the context variable ($C_{i-1}$) and the embedded mark features ($Z_j$) (up to transformations absorbed into the hazard model). As a consequence, the contrastive objective enforces that log relative hazards—and therefore relative firing rates—are linear functions of ($C_{i-1}$).

**[KEY PARAGRAPH]**
If the underlying firing rates are governed by a latent generative variable ($G$) through an arbitrary nonlinear mapping,
$\lambda(Z \mid G) = f(G, Z)$ ,
then learning a context variable ($C$) such that
$\log \lambda(Z \mid C) \approx C^\top B Z$
amounts to learning a nonlinear change of variables on ($G$) that renders prediction linear. From this perspective, ($C$) does not attempt to recover the latent state ($G$) itself, but rather a set of basis functions over ($G$) that are sufficient for linear prediction of relative rates.

This explains several empirical observations arising from the toy model. Multiple distinct bases over the same latent variable—such as Fourier-like components or sparse, localized tilings—can yield equivalent predictive performance, as they span the same function space relevant for rate prediction (the phase of the periodic generative latent). Regularization terms (e.g., L1 sparsity penalties on ($C$)) bias the model toward particular bases without altering the underlying predictive sufficiency of the representation.

<!-- ---

### Implications for interpretation of the learned context space

Taken together, these considerations clarify the interpretation of the learned context variable. The contrastive point-process objective optimizes ($C$) to be a suffucuent statistic of past information to predict future events, while the bilinear structure of the hazard enforces that this information is organized into a coordinate system in which relative firing rates are linear. Consequently, ($C$) can be interpreted as a predictive, low-dimensional embedding of latent generative structure, whose axes form a basis set tailored to the statistics of the observed neural activity.

This interpretation is particularly important for downstream analyses relating ($C$) to experimental variables. If an experimental feature can be decoded from ($C$), this implies that the feature (or a function thereof) is part of the predictive structure of the neural population, even if it is not directly observed or explicitly modeled during training. -->

Citations:

- [Predictability, Complexity, and Learning](https://arxiv.org/pdf/physics/0007070)
