from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from markovsbi.tasks.base import Task
from markovsbi.utils.prior_utils import (
    Distribution,
    GeneralDistribution,
    Normal,
    Uniform,
)


class AR1(Task):
    def __init__(
        self,
        sigma_eps: float = 1.0,
        a_min: float = -0.99,
        a_max: float = 0.99,
        x0_min: float = -10.0,
        x0_max: float = 10.0,
        T_max: int = 100,
    ):
        super().__init__("AR1")
        self.sigma_eps = sigma_eps
        self.a_min = a_min
        self.a_max = a_max
        self.x0_min = x0_min
        self.x0_max = x0_max
        self.T_max = T_max

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (1,)

    @property
    def condition_shape(self) -> tuple[int, ...]:
        return (1,)

    def get_prior(self) -> Normal:
        return Normal(jnp.array([0.0]), jnp.array([0.2]))

    def prior_x0(self) -> Normal:
        return Normal(jnp.array([0.0]), jnp.array([0.4]))

    def prior_timesteps(self) -> Uniform:
        return Uniform(jnp.array([0]), jnp.array([self.T_max]))

    def get_simulator(self) -> Callable:
        """
        Simulator for the AR(1) task.
        """

        def simulator(rng_key, theta: Array, T: int = self.T_max, x0=None, sigma_eps=1):
            """
            Simulates an AR(1) sequence given a parameter `theta` and initial state `x0`.
            :param rng_key: JAX random key.
            :param theta: Parameter `theta` for the AR(1) model.
            :param T: Length of the sequence to simulate.
            :param x0: Initial state of the sequence.
            :param sigma_eps: Standard deviation of the noise.
            :return: Simulated sequence of shape `(T, 1)`.
            """
            a = theta[0]
            key_init, key_noise = jax.random.split(rng_key)

            if x0 is None:
                x0 = jnp.squeeze(self.prior_x0().sample(key_init))
            else:
                x0 = jnp.squeeze(x0)

            noises = jax.random.normal(key_noise, (T - 1,)) * sigma_eps

            def one_step(x_t, eps_t):
                x_next = a * x_t + eps_t
                return x_next, x_next

            _, xs_tail = jax.lax.scan(one_step, x0, noises)
            xs = jnp.concatenate([jnp.array([x0]), xs_tail])
            return xs[:, None]  # shape: (T, 1)

        return simulator

    def get_data(
        self,
        key,
        num_simulations: int,
        T: int | None = None,
        informed_proposal: bool = False,
    ) -> dict[str, jnp.ndarray]:
        """
        Generates simulated data for the AR(1) task.

        For each simulation:
        1. sample theta (the AR coefficient `a`) from the prior
        2. sample a timestep t in {1, ..., T_max}
        3. sample x_0 from the x_0 prior
        4. if t == 1: start the simulator from x_0
            else:      sample x_s ~ p(x_t | a, x_0, t) and start the simulator from x_s


        :param key: JAX random key.
        :param num_simulations: Number of simulations to generate.
        :param T: Length of the sequence to simulate (starting from x_s).
        :return: Dictionary with `thetas`, `xs`, `timesteps` and `x0s`.
        """

        T = self.T_max if T is None else T
        prior = self.get_prior()
        simulator = self.get_simulator()

        key_theta, key_t, key_x0, key_marg, key_sim = jax.random.split(key, 5)

        thetas = prior.sample(key_theta, (num_simulations,))  # (N, 1)

        timesteps = jax.random.randint(
            key_t, (num_simulations,), minval=1, maxval=self.T_max + 1
        )

        x0s = self.prior_x0().sample(key_x0, (num_simulations,))  # (N, 1)

        marg_keys = jax.random.split(key_marg, num_simulations)
        sim_keys = jax.random.split(key_sim, num_simulations)

        def sample_start(k, x0, t, theta):
            a = theta[0]
            x0_s = jnp.squeeze(x0)
            # x_s ~ p(x_t | a, x_0, t)
            x_s = jnp.squeeze(self._marginal_xt(x0_s, t, a, self.sigma_eps).sample(k))
            # t == 1 -> start directly from x_0
            return jnp.where(t == 1, x0_s, x_s)

        if informed_proposal:
            x_starts = jax.vmap(sample_start)(marg_keys, x0s, timesteps, thetas)  # (N,)
        else:
            x_starts = jnp.squeeze(x0s, axis=-1)  # (N,)

        xs = jax.vmap(simulator, in_axes=(0, 0, None, 0, None))(
            sim_keys, thetas, T, x_starts, self.sigma_eps
        )  # (N, T, 1)

        return {"thetas": thetas, "xs": xs, "timesteps": timesteps, "x0s": x0s}

    def transition_log_prob(self, x_next, x_t, a, sigma_eps=1.0):
        """log p(x_{t+1} | x_t, a, sigma_eps) = log(N(x_{t+1} | a * x_t, sigma_eps))

        Args:
            x_next: The next state.
            x_t: The current state.
            a: The AR(1) coefficient.
            sigma_eps: The noise standard deviation (default: 1.0).

        Returns:
            The log probability of the next state.
        """
        return self._transition_xt(x_t, a, sigma_eps).log_prob(jnp.atleast_1d(x_next))

    def marginal_log_prob(self, x_t, a, x_0, t, sigma_eps=1.0):
        """log p(x_t | a, x_0, t) = log(N(x_t | x_0 * a^t, sqrt(1 - a^2t) / sqrt(1 - a^2)))

        Args:
            x_t: The current state.
            a: The AR(1) coefficient.
            x_0: The initial state.
            t: The time step.
            sigma_eps: The noise standard deviation (default: 1.0).

        Returns:
            The log probability of the current state.
        """
        return self._marginal_xt(x_0, t, a, sigma_eps).log_prob(jnp.atleast_1d(x_t))

    def _marginal_xt(self, x_0, t, a, sigma_eps=1.0):
        """Sample x_t from the marginal distribution p(x_t | a, x_0, t).

        Args:
            x_0: The initial state.
            t: The time step.
            a: The AR(1) coefficient.
            sigma_eps: The noise standard deviation (default: 1.0).

        Returns:
            The marginal distribution of x_t.
        """
        marginal_std = jnp.sqrt(1.0 - a ** (2 * t)) / jnp.sqrt(1.0 - a**2)
        mean = x_0 * a**t
        return Normal(jnp.atleast_1d(mean), jnp.atleast_1d(marginal_std))

    def _transition_xt(self, x_t_1, a, sigma_eps=1.0):
        """Sample x_t from the transition distribution p(x_t | x_{t-1}, a, sigma_eps).

        Args:
            x_t: The current state.
            x_t_1: The previous state.
            a: The AR(1) coefficient.
            sigma_eps: The noise standard deviation (default: 1.0).

        Returns:
            The log probability of the transition.
        """
        mean = x_t_1 * a
        return Normal(jnp.atleast_1d(mean), jnp.atleast_1d(sigma_eps))

    def sequence_log_prob(self, x, a, sigma_eps=1.0):
        """
        log p(x_{0:T} | a, sigma_eps), stationary initialization

        Args:
            x: The sequence of states.
            a: The AR(1) coefficient.
            sigma_eps: The noise standard deviation (default: 1.0).

        Returns:
            The log probability of the sequence.
        """
        x_1d = jnp.squeeze(x, axis=-1) if x.ndim >= 2 and x.shape[-1] == 1 else x
        a_scalar = jnp.squeeze(a)
        prior_x0 = self.prior_x0().log_prob(jnp.atleast_1d(x_1d[0]))
        transition_ll = jax.vmap(self.transition_log_prob, in_axes=(0, 0, None, None))(
            x_1d[1:],
            x_1d[:-1],
            a_scalar,
            sigma_eps,
        )
        return prior_x0 + transition_ll.sum(-1)

    def get_true_posterior(self, x_o: ArrayLike):
        """Exact conjugate Gaussian posterior."""
        x = jnp.squeeze(x_o)
        x_prev = x[:-1]
        x_next = x[1:]

        s_xx = jnp.sum(x_prev**2)
        s_xy = jnp.sum(x_prev * x_next)

        prior = self.get_prior()
        mu_0 = prior.mu
        sig_0 = prior.std
        sig_eps = self.sigma_eps

        prec_post = 1.0 / (sig_0**2) + s_xx / (sig_eps**2)
        var_post = 1.0 / prec_post
        std_post = jnp.sqrt(var_post)
        mu_post = var_post * (mu_0 / (sig_0**2) + s_xy / (sig_eps**2))

        return Normal(jnp.atleast_1d(mu_post), jnp.atleast_1d(std_post))
