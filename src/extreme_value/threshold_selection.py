"""
threshold_selection.py
----------------------
Data-driven threshold selection for Peaks-Over-Threshold (POT) analysis.

Two canonical graphical diagnostics for choosing the GPD threshold u:
    1. Mean Excess Function (MEF): E[X - u | X > u] should be linear in u
       for GPD-distributed exceedances.
    2. Parameter Stability Plot: Fitted xi and scale* = beta - xi * u should
       remain approximately constant above the true threshold.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from src.extreme_value.gpd import fit_gpd
from src.extreme_value.pot import extract_exceedances


def mean_excess_function(
    data: NDArray[np.float64],
    n_thresholds: int = 50,
    quantile_range: tuple[float, float] = (0.05, 0.95),
    u_min_quantile: float = 0.05,
    u_max_quantile: float = 0.95,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute the empirical mean excess function over a range of thresholds.

    The mean excess function e(u) = E[X - u | X > u] is linear in u
    for GPD-distributed exceedances. Departure from linearity indicates
    threshold mis-specification.

    Args:
        data: 1-D array of observations.
        n_thresholds: Number of threshold levels to evaluate.
        quantile_range: (low, high) quantile range for thresholds (deprecated, use u_min/u_max).
        u_min_quantile: Lower quantile bound for threshold grid. Must be in (0, 1).
        u_max_quantile: Upper quantile bound for threshold grid. Must be in (0, 1).
            Must be > u_min_quantile.

    Returns:
        Tuple of:
            - thresholds: Array of threshold values evaluated.
            - mean_excesses: Mean excess at each threshold.
            - std_excesses: Standard deviation of excesses (for confidence bands).

    Raises:
        ValueError: If quantile range is invalid.
    """
    if not (0 < u_min_quantile < u_max_quantile < 1):
        raise ValueError(
            f"u_min_quantile and u_max_quantile must satisfy "
            f"0 < u_min ({u_min_quantile}) < u_max ({u_max_quantile}) < 1."
        )

    data = np.asarray(data, dtype=np.float64).flatten()
    thresholds = np.quantile(data, np.linspace(u_min_quantile, u_max_quantile, n_thresholds))

    mean_excesses = []
    std_excesses = []
    valid_thresholds = []

    for u in thresholds:
        try:
            exc = extract_exceedances(data, float(u), tail="right")
            if len(exc) >= 2:
                mean_excesses.append(float(np.mean(exc)))
                std_excesses.append(float(np.std(exc, ddof=1)))
                valid_thresholds.append(float(u))
        except ValueError:
            continue

    return (
        np.array(valid_thresholds),
        np.array(mean_excesses),
        np.array(std_excesses),
    )


def stability_plot_data(
    data: NDArray[np.float64],
    n_thresholds: int = 40,
    u_min_quantile: float = 0.70,
    u_max_quantile: float = 0.97,
    min_exceedances: int = 10,
) -> dict[str, NDArray[np.float64]]:
    """Compute GPD parameter stability over a range of thresholds.

    Fits GPD to exceedances at each threshold and records:
        - xi (shape): should stabilize above the true threshold.
        - scale* = beta - xi * u: the modified scale, which should also stabilize.

    Args:
        data: 1-D array of observations.
        n_thresholds: Number of threshold levels.
        u_min_quantile: Starting quantile for threshold grid.
        u_max_quantile: Ending quantile for threshold grid.
        min_exceedances: Minimum number of exceedances required to fit GPD.

    Returns:
        Dictionary with:
            - ``thresholds``: Evaluated threshold values.
            - ``xi``: GPD shape estimates.
            - ``beta``: GPD scale estimates.
            - ``modified_scale``: beta - xi * u (should be stable above true threshold).
            - ``n_exceedances``: Number of exceedances at each threshold.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    thresholds = np.quantile(data, np.linspace(u_min_quantile, u_max_quantile, n_thresholds))

    xis, betas, modified_scales, n_excs, valid_u = [], [], [], [], []

    for u in thresholds:
        try:
            exc = extract_exceedances(data, float(u), tail="right")
            if len(exc) < min_exceedances:
                continue
            result = fit_gpd(exc, method="mle")
            xi = result["xi"]
            beta = result["beta"]
            xis.append(xi)
            betas.append(beta)
            modified_scales.append(beta - xi * float(u))
            n_excs.append(len(exc))
            valid_u.append(float(u))
        except Exception:
            continue

    return {
        "thresholds": np.array(valid_u),
        "xi": np.array(xis),
        "beta": np.array(betas),
        "modified_scale": np.array(modified_scales),
        "n_exceedances": np.array(n_excs, dtype=int),
    }


def plot_mean_excess(
    data: NDArray[np.float64],
    ax: plt.Axes | None = None,
    n_thresholds: int = 50,
    u_min_quantile: float = 0.05,
    u_max_quantile: float = 0.95,
    confidence: bool = True,
) -> plt.Axes:
    """Plot the empirical mean excess function.

    Args:
        data: 1-D array of observations.
        ax: Matplotlib axes. If None, a new figure is created.
        n_thresholds: Number of threshold levels.
        u_min_quantile: Lower quantile bound for threshold grid.
        u_max_quantile: Upper quantile bound for threshold grid.
        confidence: If True, draw 95% confidence bands.

    Returns:
        Matplotlib Axes with the mean excess plot.
    """
    thresholds, mean_exc, std_exc = mean_excess_function(
        data, n_thresholds=n_thresholds,
        u_min_quantile=u_min_quantile,
        u_max_quantile=u_max_quantile,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    ax.plot(thresholds, mean_exc, color="steelblue", linewidth=2, label="Mean Excess e(u)")

    if confidence and len(thresholds) > 0:
        # Approximate 95% CI using ±1.96 * std / sqrt(n_exc)
        # (n_exc decreases with u, so use approximate band)
        n_at_threshold = np.array([
            np.sum(data > u) for u in thresholds
        ])
        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            se = np.where(n_at_threshold > 1, std_exc / np.sqrt(n_at_threshold), std_exc)
        ax.fill_between(thresholds, mean_exc - 1.96 * se, mean_exc + 1.96 * se,
                        alpha=0.25, color="steelblue", label="95% CI")

    ax.set_xlabel("Threshold u", fontsize=12)
    ax.set_ylabel("Mean Excess e(u) = E[X - u | X > u]", fontsize=12)
    ax.set_title("Mean Excess Function (MEF) for Threshold Selection", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    return ax


def plot_stability(
    data: NDArray[np.float64],
    axes: tuple[plt.Axes, plt.Axes] | None = None,
    n_thresholds: int = 40,
    u_min_quantile: float = 0.70,
    u_max_quantile: float = 0.97,
) -> tuple[plt.Axes, plt.Axes]:
    """Plot GPD parameter stability over threshold range.

    Args:
        data: 1-D array of observations.
        axes: Tuple of two Matplotlib Axes. If None, creates a 1×2 figure.
        n_thresholds: Number of threshold levels.
        u_min_quantile: Lower quantile bound.
        u_max_quantile: Upper quantile bound.

    Returns:
        Tuple of two Axes: (xi_ax, scale_ax).
    """
    stability = stability_plot_data(data, n_thresholds, u_min_quantile, u_max_quantile)
    u = stability["thresholds"]
    xi = stability["xi"]
    mod_scale = stability["modified_scale"]

    if axes is None:
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    else:
        ax1, ax2 = axes

    ax1.plot(u, xi, "o-", color="teal", linewidth=1.5, markersize=4)
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Threshold u", fontsize=12)
    ax1.set_ylabel("Shape ξ (tail index)", fontsize=12)
    ax1.set_title("GPD Shape (ξ) Stability", fontsize=13)
    ax1.grid(True, alpha=0.4)

    ax2.plot(u, mod_scale, "o-", color="darkorange", linewidth=1.5, markersize=4)
    ax2.set_xlabel("Threshold u", fontsize=12)
    ax2.set_ylabel("Modified Scale β* = β − ξu", fontsize=12)
    ax2.set_title("GPD Modified Scale Stability", fontsize=13)
    ax2.grid(True, alpha=0.4)

    return ax1, ax2
