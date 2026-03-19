"""
tail_plots.py
-------------
Visualization functions for tail behavior analysis.

Provides log-log survival plots and tail QQ plots to visually assess
the heavy-tail nature of empirical data and validate fitted models.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import stats


def log_log_survival_plot(
    data: NDArray[np.float64],
    ax: plt.Axes | None = None,
    label: str = "Empirical",
    color: str = "steelblue",
    fit_power_law: bool = True,
    tail_fraction: float = 0.1,
    **kwargs: Any,
) -> plt.Axes:
    """Plot the log-log survival function (complementary CDF) to assess power-law tails.

    On a log-log scale, a Pareto/power-law tail appears as a straight line. The
    slope of this line equals -alpha (the negative tail index).

    Args:
        data: 1-D array of positive observations.
        ax: Matplotlib axes to plot on. If None, a new figure is created.
        label: Label for the empirical curve.
        color: Color for the empirical curve.
        fit_power_law: If True, fit and overlay a power-law line to the upper
            ``tail_fraction`` of the data.
        tail_fraction: Fraction of the upper tail to use for power-law fitting.
            Must be in (0, 0.5).
        **kwargs: Additional keyword arguments passed to ``ax.plot``.

    Returns:
        Matplotlib Axes object with the plot.

    Raises:
        ValueError: If data contains non-positive values.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    if np.any(data <= 0):
        raise ValueError("log_log_survival_plot requires all positive data values.")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    sorted_data = np.sort(data)
    n = len(sorted_data)
    # Survival probabilities: P(X > x) = (n - rank) / n
    survival = (n - np.arange(1, n + 1)) / n
    # Remove zero-survival points
    mask = survival > 0
    x_vals = sorted_data[mask]
    s_vals = survival[mask]

    ax.plot(np.log10(x_vals), np.log10(s_vals), "o-", color=color,
            label=label, markersize=3, linewidth=1.5, **kwargs)

    if fit_power_law:
        n_tail = max(2, int(n * tail_fraction))
        x_tail = np.log10(x_vals[-n_tail:])
        s_tail = np.log10(s_vals[-n_tail:])
        slope, intercept, r, _, _ = stats.linregress(x_tail, s_tail)
        alpha_hat = -slope  # alpha = -slope on log-log plot
        x_line = np.linspace(x_tail.min(), x_tail.max(), 100)
        ax.plot(x_line, intercept + slope * x_line, "--", color="tomato",
                label=f"Power-law fit: α ≈ {alpha_hat:.2f} (R²={r**2:.3f})")

    ax.set_xlabel("log₁₀(x)", fontsize=12)
    ax.set_ylabel("log₁₀(P(X > x))", fontsize=12)
    ax.set_title("Log-Log Survival Function (CCDF)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    return ax


def tail_qq_plot(
    data: NDArray[np.float64],
    dist: Any = None,
    dist_params: dict[str, float] | None = None,
    ax: plt.Axes | None = None,
    tail_fraction: float = 0.1,
    label: str = "Empirical",
) -> plt.Axes:
    """Generate a tail QQ plot comparing empirical vs. theoretical quantiles.

    Focuses on the upper tail region for detailed comparison of extreme quantiles.
    A good fit appears as points lying along the 45° line.

    Args:
        data: 1-D array of observations.
        dist: A scipy.stats frozen or unfrozen distribution object to compare against.
            If None, uses a standard normal for comparison.
        dist_params: Dictionary of distribution parameters (loc, scale, df, etc.).
            Used if dist is an unfrozen scipy distribution.
        ax: Matplotlib axes. If None, a new figure is created.
        tail_fraction: Upper tail fraction to plot. E.g. 0.1 shows top 10%.
        label: Label for the empirical quantile series.

    Returns:
        Matplotlib Axes object.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    n = len(data)

    if dist is None:
        dist = stats.norm

    # Select tail probabilities for the upper tail
    n_tail = max(2, int(n * tail_fraction))
    tail_probs = np.linspace(1.0 - tail_fraction, 1.0 - 1.0 / n, n_tail)

    # Empirical quantiles
    empirical_q = np.quantile(data, tail_probs)

    # Theoretical quantiles
    if dist_params:
        theoretical_q = dist.ppf(tail_probs, **dist_params)
    else:
        theoretical_q = dist.ppf(tail_probs)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(theoretical_q, empirical_q, color="steelblue", s=25,
               alpha=0.8, label=label, zorder=3)

    # 45-degree reference line
    all_vals = np.concatenate([theoretical_q, empirical_q])
    lim_min, lim_max = np.nanmin(all_vals), np.nanmax(all_vals)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1.5,
            label="Perfect fit (y=x)", zorder=2)

    ax.set_xlabel("Theoretical Quantiles", fontsize=12)
    ax.set_ylabel("Empirical Quantiles", fontsize=12)
    ax.set_title(f"Tail QQ Plot (Upper {int(tail_fraction * 100)}%)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    return ax


def hill_plot(
    data: NDArray[np.float64],
    k_min: int = 5,
    k_max: int | None = None,
    ax: plt.Axes | None = None,
    color: str = "steelblue",
    tail: str = "right",
) -> plt.Axes:
    """Plot the Hill estimator alpha(k) over a range of k values.

    The Hill plot is used to identify the stable region of the estimator,
    which guides the choice of k for tail index estimation. A plateau in
    the plot indicates the true tail index.

    Args:
        data: 1-D array of strictly positive observations.
        k_min: Minimum k to evaluate.
        k_max: Maximum k to evaluate. Defaults to n // 2.
        ax: Matplotlib axes. If None, a new figure is created.
        color: Color for the Hill estimate line.
        tail: ``"right"`` or ``"left"``.

    Returns:
        Matplotlib Axes object.
    """
    from src.tails.tail_index import hill_plot_data

    data = np.asarray(data, dtype=np.float64).flatten()
    n = len(data)

    if k_max is None:
        k_max = n // 2

    k_values, alpha_values = hill_plot_data(data, k_min=k_min, k_max=k_max, tail=tail)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    ax.plot(k_values, alpha_values, color=color, linewidth=1.5, label="Hill α(k)")
    ax.fill_between(k_values,
                    alpha_values * 0.85,
                    alpha_values * 1.15,
                    alpha=0.2, color=color)

    ax.set_xlabel("Number of Tail Order Statistics (k)", fontsize=12)
    ax.set_ylabel("Tail Index Estimate α", fontsize=12)
    ax.set_title("Hill Plot: Tail Index vs. Number of Extremes", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    # Annotate a rough middle-k estimate
    mid_k = len(k_values) // 2
    mid_alpha = alpha_values[mid_k]
    if np.isfinite(mid_alpha):
        ax.axhline(mid_alpha, color="tomato", linestyle="--", alpha=0.6,
                   label=f"α ≈ {mid_alpha:.2f} at k={k_values[mid_k]}")
        ax.legend(fontsize=10)

    return ax
