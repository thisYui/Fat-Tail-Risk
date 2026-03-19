"""
tail_metrics.py
---------------
Empirical tail probability and quantile estimation.

Provides functions to estimate extreme quantiles, tail probabilities,
and related tail statistics from data without parametric assumptions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def empirical_quantile(
    data: NDArray[np.float64],
    p: float | NDArray[np.float64],
    interpolation: str = "linear",
) -> float | NDArray[np.float64]:
    """Compute empirical quantile(s) from data.

    Args:
        data: 1-D array of observations.
        p: Probability or array of probabilities in (0, 1).
        interpolation: Interpolation method passed to ``np.quantile``.
            Options: ``"linear"``, ``"lower"``, ``"higher"``, ``"midpoint"``, ``"nearest"``.

    Returns:
        Scalar or array of empirical quantile values.

    Raises:
        ValueError: If any p not in (0, 1).
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64))

    if np.any((p_arr <= 0) | (p_arr >= 1)):
        raise ValueError("All probabilities p must be in (0, 1).")

    result = np.quantile(data, p_arr, method=interpolation)
    return float(result[0]) if np.ndim(p) == 0 else result


def tail_probability(
    data: NDArray[np.float64],
    threshold: float,
    tail: str = "right",
) -> float:
    """Estimate the empirical probability of exceeding a threshold.

    Computes P(X > threshold) for right tail or P(X < threshold) for left tail
    using the empirical CDF.

    Args:
        data: 1-D array of observations.
        threshold: The extreme threshold value.
        tail: ``"right"`` for P(X > threshold), ``"left"`` for P(X < threshold).

    Returns:
        Empirical exceedance probability in [0, 1].

    Raises:
        ValueError: If tail is not ``"right"`` or ``"left"``.
    """
    data = np.asarray(data, dtype=np.float64).flatten()

    if tail == "right":
        return float(np.mean(data > threshold))
    elif tail == "left":
        return float(np.mean(data < threshold))
    else:
        raise ValueError(f"tail must be 'right' or 'left', got '{tail}'.")


def excess_distribution(
    data: NDArray[np.float64],
    threshold: float,
) -> NDArray[np.float64]:
    """Return exceedances above a threshold (peaks over threshold).

    Computes Y = X - threshold for all X > threshold.

    Args:
        data: 1-D array of observations.
        threshold: Threshold value.

    Returns:
        Array of excess values (all >= 0).

    Raises:
        ValueError: If no exceedances are found.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    exceedances = data[data > threshold] - threshold

    if len(exceedances) == 0:
        raise ValueError(
            f"No data points exceed the threshold {threshold:.4f}. "
            "Try a lower threshold."
        )

    return exceedances


def mean_excess(
    data: NDArray[np.float64],
    threshold: float,
) -> float:
    """Compute the mean excess (mean residual life) above a threshold.

    The mean excess function e(u) = E[X - u | X > u] is linear in u for
    GPD-distributed exceedances. This property is exploited in threshold selection.

    Args:
        data: 1-D array of observations.
        threshold: Threshold u.

    Returns:
        Sample mean excess value E[X - threshold | X > threshold].

    Raises:
        ValueError: If no exceedances are found.
    """
    exceedances = excess_distribution(data, threshold)
    return float(np.mean(exceedances))


def tail_quantile(
    data: NDArray[np.float64],
    p: float,
    tail: str = "right",
) -> float:
    """Estimate an extreme quantile using the empirical distribution.

    For right tail: returns the (1-p)-th quantile (upper quantile).
    For left tail: returns the p-th quantile (lower quantile).

    Args:
        data: 1-D array of observations.
        p: Tail probability. Must be in (0, 1).
        tail: ``"right"`` for upper tail, ``"left"`` for lower tail.

    Returns:
        The extreme quantile value at probability level p.

    Raises:
        ValueError: If p is not in (0, 1) or tail is invalid.
    """
    if not (0 < p < 1):
        raise ValueError(f"p must be in (0, 1), got {p}.")

    data = np.asarray(data, dtype=np.float64).flatten()

    if tail == "right":
        return float(np.quantile(data, 1.0 - p))
    elif tail == "left":
        return float(np.quantile(data, p))
    else:
        raise ValueError(f"tail must be 'right' or 'left', got '{tail}'.")


def tail_conditional_expectation(
    data: NDArray[np.float64],
    p: float,
    tail: str = "right",
) -> float:
    """Compute the Expected Shortfall (Conditional Tail Expectation) at level p.

    ES_p = E[X | X > VaR_p] for right tail.
    Also known as CVaR or CTE. Uses empirical estimation.

    Args:
        data: 1-D array of observations.
        p: Confidence level. Must be in (0, 1).
        tail: ``"right"`` for right-tail CTE, ``"left"`` for left-tail CTE.

    Returns:
        Empirical conditional tail expectation.

    Raises:
        ValueError: If p not in (0, 1) or no tail observations found.
    """
    if not (0 < p < 1):
        raise ValueError(f"p must be in (0, 1), got {p}.")

    data = np.asarray(data, dtype=np.float64).flatten()
    var = tail_quantile(data, 1.0 - p if tail == "right" else p, tail=tail)

    if tail == "right":
        tail_data = data[data > var]
    else:
        tail_data = data[data < var]

    if len(tail_data) == 0:
        raise ValueError(
            f"No observations in the tail beyond quantile p={p}. Try a smaller p."
        )

    return float(np.mean(tail_data))


def tail_statistics(
    data: NDArray[np.float64],
    percentiles: list[float] | None = None,
) -> dict[str, float]:
    """Compute a comprehensive set of tail statistics.

    Args:
        data: 1-D array of observations.
        percentiles: List of extreme percentile levels for which to compute
            upper quantiles. Defaults to [0.90, 0.95, 0.99, 0.999].

    Returns:
        Dictionary with:
            - ``n``: Sample size.
            - ``min``, ``max``: Minimum and maximum values.
            - ``mean``, ``std``: Sample mean and standard deviation.
            - ``skewness``: Sample skewness.
            - ``kurtosis``: Sample excess kurtosis.
            - ``q{p*100}``: Empirical quantiles at each level in percentiles.
    """
    from scipy import stats

    data = np.asarray(data, dtype=np.float64).flatten()
    n = len(data)

    if percentiles is None:
        percentiles = [0.90, 0.95, 0.99, 0.999]

    result: dict[str, float] = {
        "n": float(n),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data, ddof=1)),
        "skewness": float(stats.skew(data)),
        "kurtosis": float(stats.kurtosis(data, fisher=True)),  # excess kurtosis
    }

    for p in percentiles:
        key = f"q{int(p * 100)}" if p < 1.0 else f"q{p * 100:.1f}"
        result[key] = float(np.quantile(data, p))

    return result
