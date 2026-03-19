"""
student_t.py
------------
Student-t distribution modeling.

Provides probability density, log-likelihood, and maximum likelihood
estimation (MLE) for the Student-t distribution, including joint
estimation of degrees of freedom (df) and scale.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats


def pdf(
    x: NDArray[np.float64],
    df: float,
    loc: float = 0.0,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """Compute the Student-t probability density function.

    Args:
        x: Points at which to evaluate the PDF.
        df: Degrees of freedom. Must be > 0.
        loc: Location parameter (shifts the distribution).
        scale: Scale parameter. Must be > 0.

    Returns:
        Array of PDF values at each point in x.

    Raises:
        ValueError: If df <= 0 or scale <= 0.
    """
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}.")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}.")
    x = np.asarray(x, dtype=np.float64)
    return stats.t.pdf(x, df=df, loc=loc, scale=scale)


def log_likelihood(
    data: NDArray[np.float64],
    df: float,
    loc: float = 0.0,
    scale: float = 1.0,
) -> float:
    """Compute the total log-likelihood under a Student-t distribution.

    Args:
        data: Observed data samples.
        df: Degrees of freedom. Must be > 0.
        loc: Location parameter.
        scale: Scale parameter. Must be > 0.

    Returns:
        Scalar total log-likelihood (sum over all observations).

    Raises:
        ValueError: If df <= 0 or scale <= 0.
    """
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}.")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}.")
    data = np.asarray(data, dtype=np.float64)
    return float(np.sum(stats.t.logpdf(data, df=df, loc=loc, scale=scale)))


def fit(
    data: NDArray[np.float64],
    fix_loc: float | None = None,
) -> dict[str, float]:
    """Fit a Student-t distribution to data using maximum likelihood estimation.

    Jointly estimates degrees of freedom (df), location, and scale by
    maximizing the log-likelihood via numerical optimization.

    Args:
        data: Observed data samples. Must have at least 3 observations.
        fix_loc: If provided, the location parameter is fixed at this value
            and only df and scale are optimized. Useful when the mean is known
            (e.g., fix_loc=0 for demeaned data).

    Returns:
        Dictionary with keys:
            - ``df``: Estimated degrees of freedom.
            - ``loc``: Estimated location.
            - ``scale``: Estimated scale.
            - ``log_likelihood``: Total log-likelihood at the MLE.
            - ``aic``: Akaike Information Criterion.
            - ``bic``: Bayesian Information Criterion.

    Raises:
        ValueError: If data has fewer than 3 observations.
        RuntimeError: If numerical optimization fails to converge.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    n = len(data)
    if n < 3:
        raise ValueError(f"At least 3 data points required for fitting, got {n}.")

    if fix_loc is not None:
        # Fix location, optimize df and log(scale) for numerical stability
        def neg_ll(params: np.ndarray) -> float:
            log_df, log_scale = params
            df_val = np.exp(log_df)
            scale_val = np.exp(log_scale)
            return -float(np.sum(stats.t.logpdf(data, df=df_val, loc=fix_loc, scale=scale_val)))

        x0 = [np.log(5.0), np.log(np.std(data))]
        result = optimize.minimize(neg_ll, x0, method="Nelder-Mead",
                                   options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000})
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        df_hat = float(np.exp(result.x[0]))
        scale_hat = float(np.exp(result.x[1]))
        loc_hat = float(fix_loc)
        k = 2
    else:
        # Optimize df, loc, scale jointly
        def neg_ll(params: np.ndarray) -> float:
            log_df, loc_val, log_scale = params
            df_val = np.exp(log_df)
            scale_val = np.exp(log_scale)
            return -float(np.sum(stats.t.logpdf(data, df=df_val, loc=loc_val, scale=scale_val)))

        x0 = [np.log(5.0), float(np.mean(data)), np.log(np.std(data) + 1e-8)]
        result = optimize.minimize(neg_ll, x0, method="Nelder-Mead",
                                   options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000})
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        df_hat = float(np.exp(result.x[0]))
        loc_hat = float(result.x[1])
        scale_hat = float(np.exp(result.x[2]))
        k = 3

    ll = log_likelihood(data, df_hat, loc_hat, scale_hat)
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll

    return {
        "df": df_hat,
        "loc": loc_hat,
        "scale": scale_hat,
        "log_likelihood": ll,
        "aic": aic,
        "bic": bic,
    }


def cdf(
    x: NDArray[np.float64],
    df: float,
    loc: float = 0.0,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """Compute the Student-t cumulative distribution function.

    Args:
        x: Points at which to evaluate the CDF.
        df: Degrees of freedom. Must be > 0.
        loc: Location parameter.
        scale: Scale parameter. Must be > 0.

    Returns:
        Array of CDF values at each point in x.

    Raises:
        ValueError: If df <= 0 or scale <= 0.
    """
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}.")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}.")
    x = np.asarray(x, dtype=np.float64)
    return stats.t.cdf(x, df=df, loc=loc, scale=scale)


def quantile(
    p: NDArray[np.float64],
    df: float,
    loc: float = 0.0,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """Compute the Student-t percent-point function (inverse CDF).

    Args:
        p: Probabilities. Must be in (0, 1).
        df: Degrees of freedom. Must be > 0.
        loc: Location parameter.
        scale: Scale parameter. Must be > 0.

    Returns:
        Array of quantile values.

    Raises:
        ValueError: If df <= 0, scale <= 0, or any p not in (0, 1).
    """
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}.")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}.")
    p = np.asarray(p, dtype=np.float64)
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("All probabilities p must be in (0, 1).")
    return stats.t.ppf(p, df=df, loc=loc, scale=scale)
