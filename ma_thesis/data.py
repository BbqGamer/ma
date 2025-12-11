import jax.numpy as jnp
import numpy as np


def ackley(x):
    """
    Computes the Ackley function.
    x: Input array of shape (batch_size, 2)
    https://www.sfu.ca/~ssurjano/ackley.html
    """
    # Standard Ackley parameters
    a, b, c = 20, 0.2, 2 * jnp.pi
    d = x.shape[-1]

    sum_sq = jnp.sum(x**2, axis=-1)
    sum_cos = jnp.sum(jnp.cos(c * x), axis=-1)

    term1 = -a * jnp.exp(-b * jnp.sqrt(sum_sq / d))
    term2 = -jnp.exp(sum_cos / d)

    return term1 + term2 + a + jnp.exp(1)


def bukin_function_6(x):
    """https://www.sfu.ca/~ssurjano/bukin6.html"""
    x = np.asarray(x)
    x1 = x[:, 0]
    x2 = x[:, 1]

    term1 = 100 * np.sqrt(np.abs(x2 - 0.01 * x1**2))
    term2 = 0.01 * np.abs(x1 + 10)

    return term1 + term2


def smooth(func, window_size=0.1, samples=50):
    """
    Returns a smoothed version of `func` by Gaussian kernel averaging.

    Parameters
    ----------
    func : callable
        A function f(x) where x has shape (n, d) and returns shape (n,).
    window_size : float
        Standard deviation of Gaussian kernel (smoothing strength).
    samples : int
        Number of Monte-Carlo samples used for smoothing.

    Returns
    -------
    smoothed_func : callable
        A new function that evaluates the smoothed version of `func`.
    """

    def smoothed(x):
        x = np.asarray(x)
        n, d = x.shape

        # Accumulate Monte-Carlo samples
        vals = np.zeros(n)
        for _ in range(samples):
            noise = np.random.normal(scale=window_size, size=(n, d))
            vals += func(x + noise)

        return vals / samples

    return smoothed
