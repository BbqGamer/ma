import numpy as np


def ackley(x):
    """
    Computes the Ackley function.
    x: Input array of shape (batch_size, 2)
    https://www.sfu.ca/~ssurjano/ackley.html
    """
    # Standard Ackley parameters
    a, b, c = 20, 0.2, 2 * np.pi
    d = x.shape[-1]

    sum_sq = np.sum(x**2, axis=-1)
    sum_cos = np.sum(np.cos(c * x), axis=-1)

    term1 = -a * np.exp(-b * np.sqrt(sum_sq / d))
    term2 = -np.exp(sum_cos / d)

    return term1 + term2 + a + np.exp(1)


def bukin_function_6(x):
    """https://www.sfu.ca/~ssurjano/bukin6.html"""
    x = np.asarray(x)
    x1 = x[:, 0]
    x2 = x[:, 1]

    term1 = 100 * np.sqrt(np.abs(x2 - 0.01 * x1**2))
    term2 = 0.01 * np.abs(x1 + 10)

    return term1 + term2


def levy(x):
    """
    Computes the Levy function.
    x: Input array of shape (batch_size, 2)
    https://www.sfu.ca/~ssurjano/levy.html
    """
    x = np.asarray(x)
    x1 = x[:, 0]
    x2 = x[:, 1]
    w1 = 1 + (x1 - 1) / 4
    w2 = 1 + (x2 - 1) / 4
    term1 = np.sin(np.pi * w1) ** 2
    term2 = (w1 - 1) ** 2 * (1 + 10 * np.sin(np.pi * w1 + 1) ** 2)
    term3 = (w2 - 1) ** 2 * (1 + np.sin(2 * np.pi * w2) ** 2)
    return term1 + term2 + term3


def eggholder(x):
    """
    Computes the Eggholder function.
    x: Input array of shape (batch_size, 2)
    https://www.sfu.ca/~ssurjano/egg.html
    """
    x = np.asarray(x)
    x1 = x[:, 0]
    x2 = x[:, 1]
    return -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47))) - x1 * np.sin(
        np.sqrt(np.abs(x1 - (x2 + 47)))
    )


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
