"""
tail_dependence.py
------------------
Tail dependence coefficient estimation.

Tail dependence measures the probability that two variables are simultaneously
extreme. Key coefficients:
    - Upper tail dependence: lambda_U = lim_{q->1} P(X > F_X^{-1}(q) | Y > F_Y^{-1}(q))
    - Lower tail dependence: lambda_L = lim_{q->0} P(X < F_X^{-1}(q) | Y < F_Y^{-1}(q))

For the Gaussian copula: lambda_U = lambda_L = 0 (tail independence).
For the t copula: lambda_U = lambda_L > 0 (symmetric tail dependence).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def upper_tail_dependence(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    q: float = 0.95,
) -> float:
    """Estimate the upper tail dependence coefficient.

    Empirical estimator:
        lambda_U(q) = P(U > q, V > q) / P(V > q)
                    = #{U > q and V > q} / #{V > q}

    As q -> 1, lambda_U(q) converges to the true upper tail dependence.
    In practice, q in (0.90, 0.99) is used.

    Args:
        u: Pseudo-observations for variable 1. Values in (0, 1).
        v: Pseudo-observations for variable 2. Values in (0, 1).
        q: Quantile threshold. Must be in (0, 1).

    Returns:
        Estimated upper tail dependence coefficient in [0, 1].

    Raises:
        ValueError: If u and v have different lengths or q is out of range.
    """
    u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    _validate_pseudo_observations(u, v)
    if not (0 < q < 1):
        raise ValueError(f"q must be in (0, 1), got {q}.")

    n_v_exceed = np.sum(v > q)
    if n_v_exceed == 0:
        return 0.0

    n_joint_exceed = np.sum((u > q) & (v > q))
    return float(n_joint_exceed / n_v_exceed)


def lower_tail_dependence(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    q: float = 0.05,
) -> float:
    """Estimate the lower tail dependence coefficient.

    Empirical estimator:
        lambda_L(q) = P(U < q, V < q) / P(V < q)
                    = #{U < q and V < q} / #{V < q}

    Args:
        u: Pseudo-observations for variable 1. Values in (0, 1).
        v: Pseudo-observations for variable 2. Values in (0, 1).
        q: Lower quantile threshold. Must be in (0, 1).

    Returns:
        Estimated lower tail dependence coefficient in [0, 1].

    Raises:
        ValueError: If arrays have different lengths or q is out of range.
    """
    u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    _validate_pseudo_observations(u, v)
    if not (0 < q < 1):
        raise ValueError(f"q must be in (0, 1), got {q}.")

    n_v_below = np.sum(v < q)
    if n_v_below == 0:
        return 0.0

    n_joint_below = np.sum((u < q) & (v < q))
    return float(n_joint_below / n_v_below)


def tail_dependence_profile(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    quantiles: NDArray[np.float64] | None = None,
    n_quantiles: int = 20,
) -> dict[str, NDArray[np.float64]]:
    """Compute upper and lower tail dependence over a range of quantile levels.

    Useful for visualizing how tail dependence estimates change with the
    threshold q (extrapolation to q=1 gives the true tail dependence).

    Args:
        u: Pseudo-observations for variable 1. Values in (0, 1).
        v: Pseudo-observations for variable 2. Values in (0, 1).
        quantiles: Array of q values. If None, uses a log-spaced grid in (0.5, 0.99).
        n_quantiles: Number of quantile levels if ``quantiles`` is None.

    Returns:
        Dictionary with:
            - ``q_upper``: Upper threshold quantiles.
            - ``lambda_upper``: Upper tail dependence at each q.
            - ``q_lower``: Lower threshold quantiles.
            - ``lambda_lower``: Lower tail dependence at each q.
    """
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)

    if quantiles is None:
        upper_q = np.linspace(0.50, 0.99, n_quantiles)
        lower_q = np.linspace(0.01, 0.50, n_quantiles)
    else:
        upper_q = lower_q = np.asarray(quantiles, dtype=np.float64)

    lambda_upper = np.array([upper_tail_dependence(u_arr, v_arr, float(q)) for q in upper_q])
    lambda_lower = np.array([lower_tail_dependence(u_arr, v_arr, float(q)) for q in lower_q])

    return {
        "q_upper": upper_q,
        "lambda_upper": lambda_upper,
        "q_lower": lower_q,
        "lambda_lower": lambda_lower,
    }


def theoretical_t_copula_tail_dependence(df: float, rho: float) -> float:
    """Compute the theoretical tail dependence coefficient for a t copula.

    For a bivariate t copula with correlation rho and df degrees of freedom:
        lambda = 2 * t_{df+1}(-sqrt((df+1)(1-rho)/(1+rho)))

    where t_{df+1} is the CDF of the t distribution with df+1 degrees of freedom.
    This value equals both upper and lower tail dependence (symmetric).

    Args:
        df: Degrees of freedom. Must be > 0.
        rho: Correlation coefficient. Must be in (-1, 1).

    Returns:
        Tail dependence coefficient in [0, 1].

    Raises:
        ValueError: If df <= 0 or rho not in (-1, 1).

    References:
        Joe, H. (1997). Multivariate Models and Dependence Concepts. Chapman & Hall.
    """
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}.")
    if not (-1 < rho < 1):
        raise ValueError(f"rho must be in (-1, 1), got {rho}.")

    t_arg = -np.sqrt((df + 1) * (1 - rho) / (1 + rho))
    lambda_tail = 2.0 * float(stats.t.cdf(t_arg, df=df + 1))
    return lambda_tail


def kendall_tau(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> float:
    """Compute Kendall's tau rank correlation coefficient.

    Kendall's tau measures the ordinal association between two variables
    and is related to the copula by: tau = 4 * E[C(U, V)] - 1.

    Args:
        x: 1-D array of observations for variable 1.
        y: 1-D array of observations for variable 2.

    Returns:
        Kendall's tau in [-1, 1].
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length, got {len(x)} and {len(y)}.")
    tau, _ = stats.kendalltau(x, y)
    return float(tau)


def spearman_rho(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> float:
    """Compute Spearman's rank correlation coefficient.

    Spearman's rho is related to the copula by: rho = 12 * E[C(U,V)] - 3.

    Args:
        x: 1-D array of observations for variable 1.
        y: 1-D array of observations for variable 2.

    Returns:
        Spearman's rho in [-1, 1].
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length, got {len(x)} and {len(y)}.")
    rho, _ = stats.spearmanr(x, y)
    return float(rho)


def _validate_pseudo_observations(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
) -> None:
    """Validate that u and v are valid pseudo-observations.

    Args:
        u: First pseudo-observation array.
        v: Second pseudo-observation array.

    Raises:
        ValueError: If lengths differ or values are outside (0, 1).
    """
    if len(u) != len(v):
        raise ValueError(
            f"u and v must have the same length, got {len(u)} and {len(v)}."
        )
    if np.any((u <= 0) | (u >= 1)):
        raise ValueError("u values must be in (0, 1). Use _to_pseudo_observations to transform raw data.")
    if np.any((v <= 0) | (v >= 1)):
        raise ValueError("v values must be in (0, 1). Use _to_pseudo_observations to transform raw data.")
