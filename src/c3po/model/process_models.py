import jax.numpy as jnp
import jax.nn


class ProcessModel:
    """
    Notes:
    - Models should assume parameters arrive with support on the real line
    - Models should return log-hazard and log-survival functions
    - input parameter shapes are (batch_size, n_marks, n_params)
    """

    n_params = None

    def log_survival(self, t):
        """
        Params of shape (batch_size, n_samples, n_marks, n_params)
        t is of shape (batch_size, n_samples, n_marks)
        """
        raise NotImplementedError

    def log_hazard(self, t):
        """Params and t are of shape (batch_size, n_marks, n_params) and (batch_size, n_marks) respectively"""
        raise NotImplementedError


class PoissonProcess(ProcessModel):
    """Input parameters:
    log_rate: rate of the Poisson process
    """

    n_params = 1

    def log_hazard(self, t, log_rate):
        return log_rate[..., 0]

    def log_survival(self, t, log_rate):
        return -1 * jnp.exp(log_rate[..., 0]) * t


class GompertzProcess(ProcessModel):
    n_params = 2

    def parse_params(self, params):
        # params = jnp.log(1 + jnp.exp(params))
        # print("B")
        params = jnp.log1p(jnp.exp(params))
        params = jnp.clip(params, 1e-2, 1e1)
        return params[..., 0], params[..., 1]

    def log_hazard(self, t, params):
        assert jnp.all(jnp.isfinite(params))
        assert jnp.all(jnp.isfinite(t))
        eta, beta = self.parse_params(params)
        return jnp.sum(jnp.log(eta) + beta * t, axis=1)

    def log_survival(self, t, params):
        assert jnp.all(jnp.isfinite(params))
        assert jnp.all(jnp.isfinite(t))
        eta, beta = self.parse_params(params)
        beta_t = jnp.clip(beta * t, -100, 100)
        return jnp.sum(-1 * eta / beta * (jnp.exp(beta_t) - 1), axis=1)

        # val =
        # return jnp.sum(eta / beta * jnp.clip(1 - jnp.exp(beta * t), -100, 100), axis=1)
        # return jnp.sum(eta / beta * (1 - jnp.exp(beta * t)), axis=1)


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
        # params = jnp.log(1 + jnp.exp(params))
        params = jax.nn.softplus(params)
        lam = params[..., 0]
        k = params[..., 1]
        return lam, k

    def log_hazard(self, t, params):
        lam, k = self.parse_params(params)
        p = jnp.log(k) + (k - 1) * jnp.log(t) + jnp.log(lam)
        return jnp.mean(p, axis=1)

    def log_survival(self, t, params):
        # raise NotImplementedError("not updated for sequences")
        lam, k = self.parse_params(params)
        return jnp.mean(-1 * (t * lam) ** k, axis=1)


class LogLogisticProcess(ProcessModel):
    n_params = 2

    def parse_params(self, params):
        params = jax.nn.softplus(params)
        alpha = params[..., 0]
        beta = params[..., 1]
        return alpha, beta

    def log_hazard(self, t, params):
        alpha, beta = self.parse_params(params)
        log_denom = jnp.log1p((t * alpha) ** beta)
        log_num = beta * jnp.log(alpha) + (beta - 1) * jnp.log(t)
        return jnp.mean(log_num - log_denom, axis=1)

    def log_survival(self, t, params):
        alpha, beta = self.parse_params(params)
        return jnp.mean(-1 * jnp.log1p((t * alpha) ** beta), axis=1)


distribution_dictionary = {
    "poisson": PoissonProcess,
    "gompertz": GompertzProcess,
    "weibull": WeibullProcess,
    "loglogistic": LogLogisticProcess,
}
