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
    sample_params = None

    def log_survival(self, t, params):
        """
        Params of shape (batch_size, n_samples, n_marks, n_params)
        t is of shape (batch_size, n_samples, n_marks)
        """
        raise NotImplementedError

    def log_hazard(self, t, params):
        """Params and t are of shape (batch_size, n_marks, n_params) and (batch_size, n_marks) respectively"""
        raise NotImplementedError

    def hazard(self, t, params, sum=True):
        return jnp.exp(self.log_hazard(t, params, sum=sum))

    def survival(self, t, params, sum=True):
        return jnp.exp(self.log_survival(t, params, sum=sum))


class PoissonProcess(ProcessModel):
    """Input parameters:
    log_rate: rate of the Poisson process
    """

    n_params = 1

    def log_hazard(self, t, log_rate, sum=False):
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

    def log_hazard(self, t, params, sum=False):
        assert jnp.all(jnp.isfinite(params))
        assert jnp.all(jnp.isfinite(t))
        eta, beta = self.parse_params(params)
        if sum:
            return jnp.sum(jnp.log(eta) + beta * t, axis=1)
        else:
            return jnp.log(eta) + beta * t

    def log_survival(self, t, params, sum=False):
        assert jnp.all(jnp.isfinite(params))
        assert jnp.all(jnp.isfinite(t))
        eta, beta = self.parse_params(params)
        beta_t = jnp.clip(beta * t, -100, 100)
        if sum:
            return jnp.sum(eta / beta * (jnp.exp(beta_t) - 1), axis=1)
        else:
            -1 * eta / beta * (jnp.exp(beta_t) - 1)

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

    def parse_params(self, params):
        # params = jnp.log(1 + jnp.exp(params))
        params = jax.nn.softplus(params)
        params = jnp.clip(params, 1e-5, 100)
        lam = params[..., 0]
        k = params[..., 1]
        return lam, k

    def log_hazard(self, t, params, sum=False):
        lam, k = self.parse_params(params)
        p = jnp.log(k) + (k - 1) * jnp.log(t) + jnp.log(lam)
        p = jnp.clip(p, -100, 100)
        if sum:
            return jnp.sum(p, axis=1)
        else:
            return p

    def hazard(self, t, params, sum=False):
        lam, k = self.parse_params(params)
        p = k * (t * lam) ** (k - 1) * lam
        if sum:
            return jnp.sum(p, axis=1)
        else:
            return p

    def log_survival(self, t, params, sum=False):
        # raise NotImplementedError("not updated for sequences")
        lam, k = self.parse_params(params)
        p = -1 * (t * lam) ** k
        p = jnp.clip(p, -100, 100)
        if sum:
            return jnp.sum(p, axis=1)
        else:
            return p


class MuliWeibullProcess(ProcessModel):
    n_processes = 5
    weibull = WeibullProcess()
    n_params = n_processes * WeibullProcess().n_params

    def log_hazard(self, t, params, sum=False):
        hazard = jnp.sum(
            jnp.array(
                [
                    self.weibull.log_hazard(
                        t,
                        params[
                            ...,
                            (i * WeibullProcess().n_params) : (
                                (i + 1) * WeibullProcess().n_params
                            ),
                        ],
                    )
                    for i in range(self.n_processes)
                ]
            ),
            axis=0,
        )
        return jnp.log(hazard)

    def log_survival(self, t, params, sum=False):
        log_survival = jnp.sum(
            jnp.array(
                [
                    self.weibull.log_survival(
                        t,
                        params[
                            ...,
                            (i * WeibullProcess().n_params) : (
                                (i + 1) * WeibullProcess().n_params
                            ),
                        ],
                    )
                    for i in range(self.n_processes)
                ]
            ),
            axis=0,
        )
        return log_survival


class LogLogisticProcess(ProcessModel):
    n_params = 2

    def parse_params(self, params):
        params = jax.nn.softplus(params)
        # t = jnp.clip(t, 1e-5, 10)
        # params = jnp.clip(params, 1e-5, 10)
        alpha = params[..., 0]
        alpha = jnp.clip(alpha, 1e-5, 100)
        beta = params[..., 1]
        beta = jnp.clip(beta, 1e-5, 10)
        return alpha, beta

    def hazard(self, t, params, sum=False):
        alpha, beta = self.parse_params(params)
        denom = 1 + (t * alpha) ** beta
        # num = beta * alpha**beta * t ** (beta - 1)  # TODO: check this
        num = beta * (t * alpha) ** (beta - 1) * alpha
        return num / denom

    def log_hazard(self, t, params, sum=False):
        alpha, beta = self.parse_params(params)
        log_denom = jnp.log1p((t * alpha) ** beta)
        log_num = beta * jnp.log(alpha) + (beta - 1) * jnp.log(t)
        if sum:
            return jnp.sum(log_num - log_denom, axis=1)
        else:
            return log_num - log_denom

    def log_survival(self, t, params, sum=False):
        alpha, beta = self.parse_params(params)
        if sum:
            return jnp.sum(-1 * jnp.log1p((t * alpha) ** beta), axis=1)
        else:
            # return -1 * jnp.log1p((t * alpha) ** beta)
            return -1 * jnp.log(1 + (t * alpha) ** beta)
        # return jnp.sum(-1 * jnp.log1p((t * alpha) ** beta), axis=1)


class MultiLogLogisticProcess(ProcessModel):
    n_processes = 5
    log_logistic = LogLogisticProcess()
    n_params = n_processes * LogLogisticProcess().n_params

    def log_hazard(self, t, params, sum=False):
        # params = params.reshape(-1, self.n_processes, LogLogisticProcess().n_params)
        # t = t / jnp.mean(t)
        hazard = jnp.sum(
            jnp.array(
                [
                    self.log_logistic.hazard(
                        t,
                        params[
                            ...,
                            i
                            * LogLogisticProcess().n_params : (i + 1)
                            * LogLogisticProcess().n_params,
                        ],
                        sum=sum,
                    )
                    for i in range(self.n_processes)
                ]
            ),
            axis=0,
        )
        return jnp.log(hazard)

    def log_survival(self, t, params, sum=False):
        # params = params.reshape(-1, self.n_processes, LogLogisticProcess().n_params)
        # t = t / jnp.mean(t)
        log_survival = jnp.sum(
            jnp.array(
                [
                    self.log_logistic.log_survival(
                        t,
                        params[
                            ...,
                            i
                            * LogLogisticProcess().n_params : (i + 1)
                            * LogLogisticProcess().n_params,
                        ],
                        sum=False,
                    )
                    for i in range(self.n_processes)
                ]
            ),
            axis=0,
        )
        return log_survival


class UniformDistribution(ProcessModel):
    n_params = 0
    t_max = 1000  # uniform pdf is 1/t_max

    def support_t(self, t):
        return jnp.clip(t, 0, self.t_max - 1e-5)

    def hazard(self, t, params):
        t = self.support_t(t)
        return 1 / (self.t_max - t)

    def survival(self, t, params):
        t = self.support_t(t)
        return jnp.clip(1 - t / self.t_max, 0)

    def log_hazard(self, t, params):
        t = self.support_t(t)
        return jnp.log(self.hazard(t, params))

    def log_survival(self, t, params):
        t = self.support_t(t)
        return -jnp.log(self.t_max - t)


# class LogLogisticUniform(ProcessModel):
#     n_params = 2

#     # NOTE: Composite model
#     # 1. LogLogistic
#     # 2. Uniform
#     # Reference: https://onlinelibrary.wiley.com/doi/pdf/10.1002/bimj.200210004
#     # Hazard = LogLogistic Hazard + Uniform Hazard
#     # Survival = LogLogistic Survival * Uniform Survival

#     def __init__(self):
#         self.log_logistic = LogLogisticProcess()
#         self.uniform = UniformDistribution()

#     def log_hazard(self, t, params):
#         return jnp.sum(
#             jnp.log(
#                 self.log_logistic.hazard(t, params) + self.uniform.hazard(t, params)
#             ),
#             axis=1,
#         )

#     def log_survival(self, t, params):
#         return jnp.sum(
#             self.log_logistic.log_survival(t, params, sum=False)
#             + self.uniform.log_survival(t, params),
#             axis=1,
#         )


class SilentWeightedProcess(ProcessModel):
    """Designed to allow the model to "gate" the firing rate of the underlying process.

    Args:
        base_model (ProcessModel): The underlying process model. (currently fixed but can be swapped easily)
    """

    def __init__(self):
        self.base_model = LogLogisticProcess()  # TODO: change this to accept any model
        self.n_params = self.base_model.n_params + 1

    def __call__(self, *args, **kwds):
        return self

    def parse_params(self, params):
        passed_params = params[..., :-1]
        sigma = jax.nn.sigmoid(params[..., -1] + 3)
        return sigma, passed_params

    def log_hazard(self, t, params, sum=False):
        sigma, passed_params = self.parse_params(params)
        # print("Sigma:", sigma.shape)
        # print("hazard:", self.base_model.log_hazard(t, passed_params, sum=False).shape)
        # print(
        #     "survival:", self.base_model.log_survival(t, passed_params, sum=False).shape
        # )
        numerator = (
            jnp.log(sigma)
            + self.base_model.log_hazard(t, passed_params, sum=False)
            + self.base_model.log_survival(t, passed_params, sum=False)
        )
        denominator = self.base_model.log_survival(t, params, sum=False)
        if sum:
            return jnp.sum(numerator - denominator, axis=1)
        else:
            return numerator - denominator

    def log_survival(self, t, params, sum=False):
        sigma, passed_params = self.parse_params(params)
        val = jnp.log(
            sigma * self.base_model.survival(t, passed_params, sum=False) + (1 - sigma)
        )
        if sum:
            return jnp.sum(val, axis=1)
        else:
            return val


distribution_dictionary = {
    "poisson": PoissonProcess,
    "gompertz": GompertzProcess,
    "weibull": WeibullProcess,
    "multi_weibull": MuliWeibullProcess,
    "loglogistic": LogLogisticProcess,
    # "loglogistic_uniform": LogLogisticUniform,
    "multi_loglogistic": MultiLogLogisticProcess,
    "silent_weighted": SilentWeightedProcess(),
}
