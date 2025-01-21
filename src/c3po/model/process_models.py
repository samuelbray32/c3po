import jax.numpy as jnp


class ProcessModel:
    """
    Notes:
    - Models should assume parameters arrive with support on the real line
    - Models should return log-hazard and log-survival functions
    - input parameter shapes are (batch_size, n_marks, n_params)
    """

    n_params = None

    def log_survival(self, t):
        raise NotImplementedError

    def log_hazard(self, t):
        raise NotImplementedError


class PoissonProcess(ProcessModel):
    """Input parameters:
    log_rate: rate of the Poisson process
    """

    n_params = 1

    def log_hazard(self, t, log_rate):
        return jnp.squeeze(log_rate)

    def log_survival(self, t, log_rate):
        return -1 * jnp.squeeze(jnp.sum(jnp.exp(log_rate), axis=1)) * t


class GompertzProcess(ProcessModel):
    n_params = 2

    def parse_params(self, params):
        params = jnp.log(1 + jnp.exp(params))
        params = jnp.clip(params, 1e-5, 1e1)
        return params[..., 0], params[..., 1]

    def log_hazard(self, t, params):
        eta, beta = self.parse_params(params)
        # beta * t
        return jnp.log(eta) + beta * t

    def log_survival(self, t, params):
        eta, beta = self.parse_params(params)
        return jnp.sum(eta / beta * (1 - jnp.exp(beta * t[:, None, :])), axis=1)


# Generalized Gompertz Process
# https://www.sciencedirect.com/science/article/pii/S0307904X11003118
# The generalized Gompertz distribution
# Gohary et. al. 2011


class GammaProcess(ProcessModel):

    n_params = 2

    def parse_params(self, params):
        eta = params[..., 0]
        beta = params[..., 1]
        return eta, jnp.exp(beta)

    def log_hazard(self, t, params):
        raise NotImplementedError(
            "Gamma process does not have a closed form hazard function"
        )
        eta, beta = self.parse_params(params)
        return jnp.log(eta) + beta * t

    def log_survival(self, t, params):
        eta, beta = self.parse_params(params)
        return jnp.sum(eta / beta * (1 - jnp.exp(beta * t[:, None, :])), axis=1)


class WeibullProcess(ProcessModel):
    n_params = 2

    # def parse_params(self, params):
    #     eta = jnp.exp(params[..., 0])
    #     beta = jnp.exp(params[..., 1])
    #     return eta, beta

    # def log_hazard(self, t, params):
    #     eta, beta = self.parse_params(params)
    #     return jnp.log(beta) + (beta - 1) * jnp.log(t) - (t / eta) ** beta

    # def log_survival(self, t, params):
    #     eta, beta = self.parse_params(params)
    #     return -1 * (t / eta) ** beta

    def parse_params(self, params):
        params = jnp.log(1 + jnp.exp(params))
        lam = params[..., 0]
        k = params[..., 1]
        return lam, k

    def log_hazard(self, t, params):
        lam, k = self.parse_params(params)
        return jnp.log(k) + (k - 1) * jnp.log(t) + jnp.log(lam)

    def log_survival(self, t, params):
        lam, k = self.parse_params(params)
        return jnp.sum(-1 * (t[:, None, :] * lam) ** k, axis=1)


distribution_dictionary = {
    "poisson": PoissonProcess,
    "gompertz": GompertzProcess,
    "weibull": WeibullProcess,
}
