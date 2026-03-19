"""
gpd.py
------
Generalized Pareto Distribution (GPD) fitting and inference.

The GPD is the canonical model for threshold exceedances under Extreme Value
Theory (EVT). Its three parameters are:
    - xi (shape): tail heaviness. xi > 0 => Pareto-type heavy tail.
    - beta (scale): spread of exceedances. Must be > 0.
    - mu (threshold/location): typically set to 0 for exceedances Y = X - u.

Reference:
    Balkema, A. A., & de Haan, L. (1974). Residual Life Time at Great Age.
    Annals of Probability, 2(5), 792-804.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def fit_gpd(
    exceedances: NDArray[np.float64],
    method: str = "mle",
) -> dict[str, float]:
    """Fit a Generalized Pareto Distribution to threshold exceedances.

    Args:
        exceedances: Array of positive excess values (Y = X - threshold > 0).
        method: Estimation method. ``"mle"`` uses scipy's MLE (L-BFGS-B);
            ``"pwm"`` uses probability weighted moments (moment-based, faster
            but less accurate).

    Returns:
        Dictionary with:
            - ``xi``: Estimated GPD shape (tail index). xi > 0 => heavy tail.
            - ``beta``: Estimated scale parameter.
            - ``log_likelihood``: Log-likelihood at the MLE.
            - ``aic``: Akaike Information Criterion.
            - ``bic``: Bayesian Information Criterion.
            - ``n``: Number of exceedances used.

    Raises:
        ValueError: If exceedances are not all positive, or fewer than 5.
        ValueError: If method is not ``"mle"`` or ``"pwm"``.
    """
    exceedances = np.asarray(exceedances, dtype=np.float64).flatten()
    n = len(exceedances)

    if n < 5:
        raise ValueError(f"At least 5 exceedances required for GPD fitting, got {n}.")
    if np.any(exceedances <= 0):
        raise ValueError("All exceedances must be strictly positive (Y = X - u > 0).")

    if method == "mle":
        # scipy's genpareto: parameterization is P(x; c, loc, scale)
        # c = xi (shape), loc = 0 (fixed for exceedances), scale = beta
        xi_hat, loc_hat, beta_hat = stats.genpareto.fit(exceedances, floc=0)
    elif method == "pwm":
        # Probability weighted moments estimator (Hosking & Wallis, 1987)
        xi_hat, beta_hat = _pwm_gpd(exceedances)
        loc_hat = 0.0
    else:
        raise ValueError(f"method must be 'mle' or 'pwm', got '{method}'.")

    ll = float(np.sum(stats.genpareto.logpdf(exceedances, c=xi_hat, loc=0.0, scale=beta_hat)))
    k = 2  # xi, beta
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll

    return {
        "xi": float(xi_hat),
        "beta": float(beta_hat),
        "log_likelihood": ll,
        "aic": aic,
        "bic": bic,
        "n": n,
    }


def gpd_quantile(
    p: float | NDArray[np.float64],
    xi: float,
    beta: float,
    threshold: float = 0.0,
    exceedance_rate: float = 1.0,
) -> float | NDArray[np.float64]:
    """Compute extreme quantiles using a fitted GPD model.

    Computes the quantile at probability level 1 - p (i.e., P(X > q) = p)
    using the GPD fitted to exceedances above threshold u.

    Formula:
        q(p) = u + (beta / xi) * [ (p / exceedance_rate)^(-xi) - 1 ]  for xi != 0
        q(p) = u - beta * log(p / exceedance_rate)                      for xi = 0

    Args:
        p: Exceedance probability (tail probability). P(X > q) = p.
            Must be in (0, exceedance_rate).
        xi: GPD shape parameter.
        beta: GPD scale parameter. Must be > 0.
        threshold: The original threshold u.
        exceedance_rate: The fraction of data exceeding the threshold (P(X > u)).

    Returns:
        Extreme quantile value(s) q such that P(X > q) = p.

    Raises:
        ValueError: If beta <= 0 or p >= exceedance_rate.
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}.")

    p_arr = np.asarray(p, dtype=np.float64)
    scalar = p_arr.ndim == 0
    p_arr = np.atleast_1d(p_arr)

    if np.any(p_arr >= exceedance_rate):
        raise ValueError(
            f"p must be < exceedance_rate={exceedance_rate:.4f} (threshold exceedance probability)."
        )

    ratio = p_arr / exceedance_rate

    if abs(xi) < 1e-10:  # xi ≈ 0: exponential case
        quantiles = threshold - beta * np.log(ratio)
    else:
        quantiles = threshold + (beta / xi) * (ratio ** (-xi) - 1.0)

    return float(quantiles[0]) if scalar else quantiles


def gpd_tail_probability(
    x: float | NDArray[np.float64],
    xi: float,
    beta: float,
    threshold: float,
    exceedance_rate: float,
) -> float | NDArray[np.float64]:
    """Compute tail probability P(X > x) using the fitted GPD.

    Args:
        x: Value(s) at which to compute tail probability.
        xi: GPD shape.
        beta: GPD scale.
        threshold: Original threshold u.
        exceedance_rate: Fraction of data exceeding threshold.

    Returns:
        P(X > x) for each x.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    excess = x_arr - threshold
    # P(X > x) = exceedance_rate * P(Y > x - u) where Y ~ GPD(xi, beta)
    survival = 1.0 - stats.genpareto.cdf(excess, c=xi, loc=0.0, scale=beta)
    return float(exceedance_rate * survival) if np.ndim(x) == 0 else exceedance_rate * survival


def _pwm_gpd(exceedances: NDArray[np.float64]) -> tuple[float, float]:
    """Estimate GPD parameters using Probability Weighted Moments (PWM).

    Args:
        exceedances: Positive excess values.

    Returns:
        Tuple of (xi_hat, beta_hat).

    References:
        Hosking, J. R. M., & Wallis, J. R. (1987). Parameter and Quantile
        Estimation for the Generalized Pareto Distribution.
        Technometrics, 29(3), 339-349.
    """
    y = np.sort(exceedances)
    n = len(y)
    # First two PWM estimators
    b0 = np.mean(y)
    # b1 = (1/n) * sum_{i=1}^{n} (i-1)/(n-1) * y_i
    weights = (np.arange(0, n) / (n - 1))
    b1 = float(np.dot(weights, y) / n)

    # Hosking & Wallis formulas
    beta_hat = 2.0 * b0 * b1 / (b0 - 2.0 * b1)
    xi_hat = 2.0 - b0 / (b0 - 2.0 * b1)

    return xi_hat, beta_hat
