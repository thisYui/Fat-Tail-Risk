"""
metrics.py
----------
Evaluation metrics for distribution fitting and tail estimation.

Quantifies the accuracy of fitted models by comparing predicted vs.
empirical quantiles, tail probabilities, and other distributional properties.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def quantile_error(
    data: NDArray[np.float64],
    quantile_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    probabilities: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64] | float]:
    """Compute errors between empirical and model-predicted quantiles.

    Args:
        data: 1-D observed data.
        quantile_fn: Callable mapping probability levels to model quantiles.
            Example: ``lambda p: scipy.stats.t.ppf(p, df=3, loc=0, scale=1)``.
        probabilities: Probability levels to evaluate. Defaults to
            [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99].

    Returns:
        Dictionary with:
            - ``probabilities``: Probability levels evaluated.
            - ``empirical_quantiles``: Sample quantiles.
            - ``model_quantiles``: Model-predicted quantiles.
            - ``absolute_errors``: |empirical - model| at each level.
            - ``mean_absolute_error``: Mean of absolute errors.
            - ``max_absolute_error``: Maximum absolute error.
            - ``relative_errors``: |empirical - model| / |model| (safe).
    """
    data = np.asarray(data, dtype=np.float64).flatten()

    if probabilities is None:
        probabilities = np.array([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    else:
        probabilities = np.asarray(probabilities, dtype=np.float64)

    empirical_q = np.quantile(data, probabilities)
    model_q = quantile_fn(probabilities)
    model_q = np.asarray(model_q, dtype=np.float64)

    abs_errors = np.abs(empirical_q - model_q)
    # Relative error = |err| / |model|, safe against near-zero denominators
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_errors = np.where(np.abs(model_q) > 1e-10, abs_errors / np.abs(model_q), np.nan)

    return {
        "probabilities": probabilities,
        "empirical_quantiles": empirical_q,
        "model_quantiles": model_q,
        "absolute_errors": abs_errors,
        "mean_absolute_error": float(np.mean(abs_errors)),
        "max_absolute_error": float(np.max(abs_errors)),
        "relative_errors": rel_errors,
    }


def tail_quantile_error(
    data: NDArray[np.float64],
    quantile_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    tail_probabilities: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64] | float]:
    """Compute errors specifically in the tail region.

    Focuses on extreme probabilities relevant for tail risk analysis.

    Args:
        data: 1-D observed data.
        quantile_fn: Callable mapping probabilities to model quantiles.
        tail_probabilities: Extreme probability levels. Defaults to
            [0.95, 0.975, 0.99, 0.995, 0.999].

    Returns:
        Same structure as ``quantile_error`` but for tail probabilities only.
    """
    if tail_probabilities is None:
        tail_probabilities = np.array([0.95, 0.975, 0.99, 0.995, 0.999])

    return quantile_error(data, quantile_fn, tail_probabilities)


def tail_probability_error(
    data: NDArray[np.float64],
    cdf_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    thresholds: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64] | float]:
    """Compare empirical vs. model tail probabilities P(X > threshold).

    Args:
        data: 1-D observed data.
        cdf_fn: Callable mapping values to model CDF probabilities F(x).
        thresholds: Values at which to compare P(X > x). Defaults to
            the 90th through 99.9th empirical quantiles.

    Returns:
        Dictionary with:
            - ``thresholds``: Threshold values used.
            - ``empirical_tail_probs``: Fraction of data exceeding each threshold.
            - ``model_tail_probs``: ``1 - cdf_fn(threshold)`` at each threshold.
            - ``absolute_errors``: Absolute difference.
            - ``mean_absolute_error``: Mean absolute error.
            - ``log_ratio_errors``: log(model_prob / empirical_prob) for log-scale assessment.
    """
    data = np.asarray(data, dtype=np.float64).flatten()

    if thresholds is None:
        thresholds = np.quantile(data, [0.90, 0.925, 0.95, 0.975, 0.99, 0.995, 0.999])

    thresholds = np.asarray(thresholds, dtype=np.float64)
    empirical_probs = np.array([float(np.mean(data > u)) for u in thresholds])
    model_probs = 1.0 - np.asarray(cdf_fn(thresholds), dtype=np.float64)

    abs_errors = np.abs(empirical_probs - model_probs)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratios = np.where(
            (model_probs > 0) & (empirical_probs > 0),
            np.log(model_probs / empirical_probs),
            np.nan,
        )

    return {
        "thresholds": thresholds,
        "empirical_tail_probs": empirical_probs,
        "model_tail_probs": model_probs,
        "absolute_errors": abs_errors,
        "mean_absolute_error": float(np.nanmean(abs_errors)),
        "log_ratio_errors": log_ratios,
    }


def wasserstein_distance(
    data_1: NDArray[np.float64],
    data_2: NDArray[np.float64],
) -> float:
    """Compute the 1-Wasserstein (Earth Mover's) distance between two samples.

    W_1 = integral |F_1(x) - F_2(x)| dx

    For 1-D distributions, this equals the L1 norm between sorted samples
    (or equivalently, the mean absolute difference of matched quantiles).

    Args:
        data_1: First sample array.
        data_2: Second sample array.

    Returns:
        Wasserstein-1 distance (non-negative).
    """
    from scipy.stats import wasserstein_distance as scipy_wd
    d1 = np.asarray(data_1, dtype=np.float64).flatten()
    d2 = np.asarray(data_2, dtype=np.float64).flatten()
    return float(scipy_wd(d1, d2))


def kolmogorov_smirnov_distance(
    data: NDArray[np.float64],
    cdf_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> float:
    """Compute the KS distance (maximum CDF deviation) as a scalar metric.

    Args:
        data: Observed data.
        cdf_fn: Model CDF callable.

    Returns:
        KS statistic D = max |F_n(x) - F(x)|.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    ks_stat, _ = stats.kstest(data, cdf_fn)
    return float(ks_stat)
