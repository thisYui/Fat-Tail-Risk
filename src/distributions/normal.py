"""
normal.py
---------
Gaussian (Normal) distribution modeling.

Provides probability density, log-likelihood, and maximum likelihood
estimation (MLE) for the Normal distribution.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def pdf(
    x: NDArray[np.float64],
    mu: float,
    sigma: float,
) -> NDArray[np.float64]:
    """Compute the Gaussian probability density function.

    Args:
        x: Points at which to evaluate the PDF.
        mu: Mean of the distribution.
        sigma: Standard deviation. Must be > 0.

    Returns:
        Array of PDF values at each point in x.

    Raises:
        ValueError: If sigma <= 0.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    x = np.asarray(x, dtype=np.float64)
    return stats.norm.pdf(x, loc=mu, scale=sigma)


def log_likelihood(
    data: NDArray[np.float64],
    mu: float,
    sigma: float,
) -> float:
    """Compute the total log-likelihood of data under a Gaussian distribution.

    Args:
        data: Observed data samples.
        mu: Mean of the distribution.
        sigma: Standard deviation. Must be > 0.

    Returns:
        Scalar total log-likelihood value (sum over all observations).

    Raises:
        ValueError: If sigma <= 0.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    data = np.asarray(data, dtype=np.float64)
    return float(np.sum(stats.norm.logpdf(data, loc=mu, scale=sigma)))


def fit(
    data: NDArray[np.float64],
) -> dict[str, float]:
    """Fit a Gaussian distribution to data using maximum likelihood estimation.

    For the Gaussian, MLE estimates are simply the sample mean and standard
    deviation (with the MLE denominator n, not n-1).

    Args:
        data: Observed data samples. Must have at least 2 observations.

    Returns:
        Dictionary with keys:
            - ``mu``: MLE estimate of the mean.
            - ``sigma``: MLE estimate of the standard deviation.
            - ``log_likelihood``: Total log-likelihood at the MLE.
            - ``aic``: Akaike Information Criterion (2k - 2LL).
            - ``bic``: Bayesian Information Criterion (k*log(n) - 2LL).

    Raises:
        ValueError: If data has fewer than 2 observations.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    n = len(data)
    if n < 2:
        raise ValueError(f"At least 2 data points required for fitting, got {n}.")

    mu_hat = float(np.mean(data))
    # MLE uses n denominator (not n-1)
    sigma_hat = float(np.std(data, ddof=0))

    if sigma_hat == 0.0:
        raise ValueError("All data values are identical; cannot fit a Normal distribution.")

    ll = log_likelihood(data, mu_hat, sigma_hat)
    k = 2  # number of parameters: mu, sigma
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll

    return {
        "mu": mu_hat,
        "sigma": sigma_hat,
        "log_likelihood": ll,
        "aic": aic,
        "bic": bic,
    }


def cdf(
    x: NDArray[np.float64],
    mu: float,
    sigma: float,
) -> NDArray[np.float64]:
    """Compute the Gaussian cumulative distribution function.

    Args:
        x: Points at which to evaluate the CDF.
        mu: Mean of the distribution.
        sigma: Standard deviation. Must be > 0.

    Returns:
        Array of CDF values (probabilities) at each point in x.

    Raises:
        ValueError: If sigma <= 0.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    x = np.asarray(x, dtype=np.float64)
    return stats.norm.cdf(x, loc=mu, scale=sigma)


def quantile(
    p: NDArray[np.float64],
    mu: float,
    sigma: float,
) -> NDArray[np.float64]:
    """Compute the Gaussian percent-point function (inverse CDF).

    Args:
        p: Probabilities at which to evaluate the quantile function.
            Values must be in (0, 1).
        mu: Mean of the distribution.
        sigma: Standard deviation. Must be > 0.

    Returns:
        Array of quantile values corresponding to probabilities p.

    Raises:
        ValueError: If sigma <= 0 or any p not in (0, 1).
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    p = np.asarray(p, dtype=np.float64)
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("All probabilities p must be in (0, 1).")
    return stats.norm.ppf(p, loc=mu, scale=sigma)
