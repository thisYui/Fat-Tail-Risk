"""
tail_index.py
-------------
Tail index estimation for heavy-tailed distributions.

The tail index (also called Pareto exponent or Hill exponent) characterizes
how heavy the right tail of a distribution is. Key estimators:
    - Hill estimator: The classical and most widely used estimator.
    - Pickands estimator: A robust alternative, less sensitive to threshold choice.
    - Moments estimator: Uses ratio of power moments for a bias-corrected estimate.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def estimate_tail_index(
    data: NDArray[np.float64],
    k: int,
    tail: str = "right",
) -> float:
    """Estimate the tail index using the Hill estimator.

    The Hill estimator uses the k largest order statistics to estimate
    the Pareto exponent alpha (tail index), where alpha = 1 / xi
    (xi is the GPD shape parameter).

    Hill estimator formula:
        alpha_hat = k / sum_{i=1}^{k} [ log(X_(n-i+1)) - log(X_(n-k)) ]

    where X_(1) <= X_(2) <= ... <= X_(n) are order statistics.

    Args:
        data: 1-D array of observations. Must be positive for a right-tail estimate.
            For left-tail analysis, the sign flip is handled internally.
        k: Number of upper order statistics to use. Must satisfy 1 < k < n.
            A common heuristic is k ~ sqrt(n) or k ~ n^(2/3).
        tail: Which tail to analyze. ``"right"`` uses the largest values;
            ``"left"`` negates the data and analyzes the right tail of -data.

    Returns:
        Estimated tail index alpha (> 0). Larger alpha => lighter tail.

    Raises:
        ValueError: If k is out of range, data has insufficient length, or
            data contains non-positive values for right-tail estimation.

    References:
        Hill, B. M. (1975). A Simple General Approach to Inference About the
        Tail of a Distribution. Annals of Statistics, 3(5), 1163-1174.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    n = len(data)

    if tail == "left":
        # For left-tail analysis, we negate and work with the right tail of -data
        data = -data

    if np.any(data <= 0):
        raise ValueError(
            "Hill estimator requires strictly positive data. "
            "For right-tail analysis, use only positive observations (e.g., losses). "
            "For left-tail analysis, pass tail='left'."
        )

    if not (1 < k < n):
        raise ValueError(
            f"k must satisfy 1 < k < n (n={n}), got k={k}. "
            f"Recommended: k ~ sqrt(n) = {int(np.sqrt(n))}."
        )

    # Sort ascending and take the k largest values
    sorted_data = np.sort(data)
    # X_(n-k) is the (k+1)-th largest value (0-indexed: sorted_data[n-k-1])
    x_threshold = sorted_data[n - k - 1]
    x_top_k = sorted_data[n - k:]  # k largest values: X_(n-k+1), ..., X_(n)

    # Hill estimator: 1/alpha = (1/k) * sum of log(X_i / X_threshold)
    log_ratios = np.log(x_top_k) - np.log(x_threshold)
    xi_hat = float(np.mean(log_ratios))  # shape parameter xi = 1/alpha

    if xi_hat <= 0:
        raise RuntimeError(
            f"Hill estimator produced non-positive xi={xi_hat:.4f}, which implies "
            "a bounded or thin-tailed distribution. Check your data and k choice."
        )

    return 1.0 / xi_hat  # Return alpha (tail index)


def hill_estimator(
    data: NDArray[np.float64],
    k: int,
    tail: str = "right",
) -> float:
    """Alias for estimate_tail_index using the Hill estimator.

    See ``estimate_tail_index`` for full documentation.

    Args:
        data: 1-D array of strictly positive observations.
        k: Number of upper order statistics to use.
        tail: ``"right"`` or ``"left"``.

    Returns:
        Estimated tail index alpha.
    """
    return estimate_tail_index(data, k, tail)


def hill_plot_data(
    data: NDArray[np.float64],
    k_min: int = 5,
    k_max: int | None = None,
    tail: str = "right",
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Compute Hill estimates over a range of k values for the Hill plot.

    The Hill plot shows alpha_hat(k) vs. k. A stable region (plateau) in the
    Hill plot indicates the optimal k and the true tail index.

    Args:
        data: 1-D array of strictly positive observations.
        k_min: Minimum k to compute. Must be > 1.
        k_max: Maximum k to compute. Defaults to n//2.
        tail: ``"right"`` or ``"left"``.

    Returns:
        Tuple of:
            - k_values: Array of k values evaluated.
            - alpha_values: Corresponding Hill estimates.

    Raises:
        ValueError: If k_min < 2 or k_max >= n.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    n = len(data)

    if k_max is None:
        k_max = n // 2
    if k_min < 2:
        raise ValueError(f"k_min must be >= 2, got {k_min}.")
    if k_max >= n:
        raise ValueError(f"k_max must be < n={n}, got {k_max}.")

    k_values = np.arange(k_min, k_max + 1, dtype=np.int64)
    alpha_values = np.empty(len(k_values))

    for i, k in enumerate(k_values):
        try:
            alpha_values[i] = estimate_tail_index(data, int(k), tail)
        except Exception:
            alpha_values[i] = np.nan

    return k_values, alpha_values


def pickands_estimator(
    data: NDArray[np.float64],
    k: int,
    tail: str = "right",
) -> float:
    """Estimate the tail shape parameter xi using the Pickands estimator.

    The Pickands estimator is:
        xi_hat = (1 / log(2)) * log( (X_(n-k+1) - X_(n-2k+1)) / (X_(n-2k+1) - X_(n-4k+1)) )

    It estimates xi directly (GPD shape) rather than alpha = 1/xi.
    This estimator requires n >= 4k.

    Args:
        data: 1-D array of observations.
        k: Integer such that 4k < n.
        tail: ``"right"`` or ``"left"``.

    Returns:
        Estimated GPD shape parameter xi.

    Raises:
        ValueError: If 4k >= n.

    References:
        Pickands, J. (1975). Statistical Inference Using Extreme Order Statistics.
        Annals of Statistics, 3(1), 119-131.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    n = len(data)

    if tail == "left":
        data = -data

    if 4 * k >= n:
        raise ValueError(
            f"Pickands estimator requires 4k < n. Got k={k}, n={n} => 4k={4*k} >= {n}."
        )

    sorted_data = np.sort(data)  # ascending
    # Use 1-based order statistics (ascending indexing from the right)
    x1 = sorted_data[n - k - 1]       # X_(n-k)
    x2 = sorted_data[n - 2 * k - 1]   # X_(n-2k)
    x4 = sorted_data[n - 4 * k - 1]   # X_(n-4k)

    numerator = x1 - x2
    denominator = x2 - x4

    if denominator <= 0 or numerator <= 0:
        raise RuntimeError(
            "Pickands estimator encountered non-positive differences. "
            "Check data quality and choice of k."
        )

    xi_hat = np.log(numerator / denominator) / np.log(2)
    return float(xi_hat)


def moments_estimator(
    data: NDArray[np.float64],
    k: int,
    tail: str = "right",
) -> float:
    """Estimate xi using the Dekkers-Einmahl-de Haan moments estimator.

    The moments estimator is:
        M_1 = (1/k) * sum_{i=1}^{k} log(X_(n-i+1) / X_(n-k))
        M_2 = (1/k) * sum_{i=1}^{k} (log(X_(n-i+1) / X_(n-k)))^2
        xi_hat = M_1 + 1 - 0.5 / (1 - M_1^2 / M_2)

    This estimator works for all values of xi (unlike Hill which requires xi > 0).

    Args:
        data: 1-D array of observations.
        k: Number of upper order statistics.
        tail: ``"right"`` or ``"left"``.

    Returns:
        Estimated tail shape parameter xi real-valued.

    Raises:
        ValueError: If k is out of bounds.

    References:
        Dekkers, A. L. M., Einmahl, J. H. J., & de Haan, L. (1989).
        A moment estimator for the index of an extreme-value distribution.
        Annals of Statistics, 17(4), 1833-1855.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    n = len(data)

    if tail == "left":
        data = -data

    if not (1 < k < n):
        raise ValueError(f"k must satisfy 1 < k < n={n}, got k={k}.")

    sorted_data = np.sort(data)
    x_threshold = sorted_data[n - k - 1]
    x_top_k = sorted_data[n - k:]

    log_ratios = np.log(x_top_k) - np.log(x_threshold)
    m1 = float(np.mean(log_ratios))
    m2 = float(np.mean(log_ratios ** 2))

    if m2 <= 0:
        raise RuntimeError("M2 moment is zero or negative; check data and k.")

    xi_hat = m1 + 1.0 - 0.5 / (1.0 - m1**2 / m2)
    return float(xi_hat)
