"""
pot.py
------
Peaks-Over-Threshold (POT) extraction for Extreme Value Theory.

The POT method selects observations that exceed a given threshold u,
then models the threshold exceedances using the Generalized Pareto
Distribution (GPD).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def extract_exceedances(
    data: NDArray[np.float64],
    threshold: float,
    tail: str = "right",
) -> NDArray[np.float64]:
    """Extract threshold exceedances (peaks over threshold).

    Computes Y_i = X_i - threshold for all X_i > threshold (right tail) or
    Y_i = threshold - X_i for all X_i < threshold (left tail).

    Args:
        data: 1-D array of observations.
        threshold: The threshold value u.
        tail: ``"right"`` for upper-tail exceedances, ``"left"`` for lower-tail.

    Returns:
        Array of positive exceedance values (Y_i > 0).

    Raises:
        ValueError: If no exceedances are found at the given threshold.
    """
    data = np.asarray(data, dtype=np.float64).flatten()

    if tail == "right":
        exceedances = data[data > threshold] - threshold
    elif tail == "left":
        exceedances = threshold - data[data < threshold]
    else:
        raise ValueError(f"tail must be 'right' or 'left', got '{tail}'.")

    if len(exceedances) == 0:
        raise ValueError(
            f"No exceedances found at threshold={threshold:.4f} in the {tail} tail. "
            "Try a lower (right tail) or higher (left tail) threshold."
        )

    return exceedances


def pot_summary(
    data: NDArray[np.float64],
    threshold: float,
    tail: str = "right",
) -> dict[str, float | int | NDArray[np.float64]]:
    """Summarize the peaks-over-threshold dataset.

    Args:
        data: 1-D array of observations.
        threshold: Threshold u.
        tail: ``"right"`` or ``"left"``.

    Returns:
        Dictionary with:
            - ``threshold``: The threshold used.
            - ``n_total``: Total number of observations.
            - ``n_exceedances``: Number of exceedances.
            - ``exceedance_rate``: Fraction of observations that exceed the threshold.
            - ``exceedances``: Array of exceedance values.
            - ``mean_excess``: Mean of exceedances (mean excess function at threshold).
            - ``max_excess``: Maximum exceedance.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    exceedances = extract_exceedances(data, threshold, tail)
    n_total = len(data)
    n_exc = len(exceedances)

    return {
        "threshold": threshold,
        "n_total": n_total,
        "n_exceedances": n_exc,
        "exceedance_rate": n_exc / n_total,
        "exceedances": exceedances,
        "mean_excess": float(np.mean(exceedances)),
        "max_excess": float(np.max(exceedances)),
    }


def threshold_range_analysis(
    data: NDArray[np.float64],
    thresholds: NDArray[np.float64] | None = None,
    n_thresholds: int = 30,
    quantile_range: tuple[float, float] = (0.80, 0.99),
    tail: str = "right",
) -> list[dict[str, float | int]]:
    """Analyze exceedance statistics across a range of thresholds.

    Useful for understanding how the number of exceedances and mean excess
    change as a function of threshold. Feeds into threshold selection tools.

    Args:
        data: 1-D array of observations.
        thresholds: Explicit threshold values to evaluate. If None, uses
            quantile-spaced thresholds.
        n_thresholds: Number of threshold levels if ``thresholds`` is None.
        quantile_range: (low, high) quantile range for auto-generated thresholds.
        tail: ``"right"`` or ``"left"``.

    Returns:
        List of dictionaries, each containing:
            - ``threshold``: The threshold value.
            - ``n_exceedances``: Number of exceedances.
            - ``exceedance_rate``: Fraction exceeding.
            - ``mean_excess``: Mean excess at this threshold.
    """
    data = np.asarray(data, dtype=np.float64).flatten()

    if thresholds is None:
        q_low, q_high = quantile_range
        thresholds = np.quantile(data, np.linspace(q_low, q_high, n_thresholds))

    results = []
    for u in thresholds:
        try:
            summary = pot_summary(data, float(u), tail)
            results.append({
                "threshold": summary["threshold"],
                "n_exceedances": summary["n_exceedances"],
                "exceedance_rate": summary["exceedance_rate"],
                "mean_excess": summary["mean_excess"],
            })
        except ValueError:
            # No exceedances at this threshold; skip
            continue

    return results
