"""
qq.py
-----
QQ plot generators for theoretical vs. empirical quantile comparison.

QQ plots are essential for visually assessing whether data follows a
specified theoretical distribution. Points lying on the 45° line indicate
a good fit; deviations reveal systematic differences.
"""

from __future__ import annotations

from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import stats


def qq_plot(
    data: NDArray[np.float64],
    dist: Any = None,
    dist_params: dict[str, float] | None = None,
    n_quantiles: int | None = None,
    ax: plt.Axes | None = None,
    label: str = "Data",
    color: str = "steelblue",
    show_confidence: bool = True,
    alpha_level: float = 0.05,
) -> plt.Axes:
    """Generate a QQ plot comparing empirical to theoretical quantiles.

    Args:
        data: 1-D array of observed values.
        dist: Scipy frozen or unfrozen distribution. Defaults to ``scipy.stats.norm``.
        dist_params: Keyword arguments for ``dist.ppf`` if dist is unfrozen
            (e.g., ``{"df": 3, "loc": 0, "scale": 1}``).
        n_quantiles: Number of equally-spaced quantile probabilities to plot.
            Defaults to min(len(data), 200).
        ax: Matplotlib axes. If None, a new figure is created.
        label: Label for the data scatter points.
        color: Color for the scatter points.
        show_confidence: If True, draw Kolmogorov-Smirnov confidence bands.
        alpha_level: Significance level for confidence bands.

    Returns:
        Matplotlib Axes with the QQ plot.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    n = len(data)

    if dist is None:
        dist = stats.norm
        dist_params = {}

    if dist_params is None:
        dist_params = {}

    if n_quantiles is None:
        n_quantiles = min(n, 200)

    # Probability levels (avoid 0 and 1)
    probs = np.linspace(1 / (n + 1), n / (n + 1), n_quantiles)

    empirical_q = np.quantile(data, probs)
    theoretical_q = dist.ppf(probs, **dist_params)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(theoretical_q, empirical_q, s=20, color=color, alpha=0.8,
               label=label, zorder=3)

    # 45-degree reference line
    all_vals = np.concatenate([theoretical_q, empirical_q])
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    ax.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=1.5,
            label="Perfect fit (y=x)", zorder=2)

    if show_confidence:
        # Kolmogorov-Smirnov confidence band width
        c_alpha = np.sqrt(-np.log(alpha_level / 2) / (2 * n))
        ax.plot([vmin, vmax], [vmin + c_alpha, vmax + c_alpha], "k:",
                alpha=0.5, linewidth=1.0, label=f"{int((1-alpha_level)*100)}% CI band")
        ax.plot([vmin, vmax], [vmin - c_alpha, vmax - c_alpha], "k:",
                alpha=0.5, linewidth=1.0)

    ax.set_xlabel("Theoretical Quantiles", fontsize=12)
    ax.set_ylabel("Empirical Quantiles", fontsize=12)
    ax.set_title("QQ Plot: Empirical vs. Theoretical", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    return ax


def pp_plot(
    data: NDArray[np.float64],
    cdf: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ax: plt.Axes | None = None,
    label: str = "Empirical",
    color: str = "teal",
) -> plt.Axes:
    """Generate a PP plot (probability-probability plot).

    Plots the empirical CDF against the theoretical CDF at observed data points.
    Points lying on the 45° line indicate a good fit.

    Args:
        data: 1-D array of observed values.
        cdf: Callable returning theoretical CDF probabilities.
        ax: Matplotlib axes. If None, creates a new figure.
        label: Label for the empirical series.
        color: Color for the scatter points.

    Returns:
        Matplotlib Axes with the PP plot.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    n = len(data)

    sorted_data = np.sort(data)
    empirical_probs = np.arange(1, n + 1) / (n + 1)  # continuity correction
    theoretical_probs = cdf(sorted_data)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(theoretical_probs, empirical_probs, s=20, color=color,
               alpha=0.8, label=label, zorder=3)
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Perfect fit", zorder=2)

    ax.set_xlabel("Theoretical Probability F(x)", fontsize=12)
    ax.set_ylabel("Empirical Probability F_n(x)", fontsize=12)
    ax.set_title("PP Plot: Empirical vs. Theoretical Probabilities", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return ax


def qq_residuals(
    data: NDArray[np.float64],
    dist: Any,
    dist_params: dict[str, float] | None = None,
    n_quantiles: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute QQ residuals (empirical - theoretical quantiles).

    Args:
        data: 1-D array of observed values.
        dist: Scipy distribution.
        dist_params: Parameters for the distribution.
        n_quantiles: Number of quantile levels.

    Returns:
        Tuple of:
            - probs: Quantile probability levels.
            - residuals: empirical_q - theoretical_q at each level.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    n = len(data)

    if dist_params is None:
        dist_params = {}

    if n_quantiles is None:
        n_quantiles = min(n, 200)

    probs = np.linspace(1 / (n + 1), n / (n + 1), n_quantiles)
    empirical_q = np.quantile(data, probs)
    theoretical_q = dist.ppf(probs, **dist_params)

    return probs, empirical_q - theoretical_q


def multi_qq_plot(
    data: NDArray[np.float64],
    distributions: list[dict[str, Any]],
    n_quantiles: int = 100,
    figsize: tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Plot multiple QQ plots side-by-side for distribution comparison.

    Args:
        data: 1-D array of observed values.
        distributions: List of dicts specifying distributions to compare.
            Each dict must have:
                - ``dist``: Scipy distribution object.
                - ``params``: Dict of parameters for dist.ppf.
                - ``label``: Label string.
        n_quantiles: Number of quantile levels.
        figsize: Figure size.

    Returns:
        Matplotlib Figure containing the plots.
    """
    n_dists = len(distributions)
    fig, axes = plt.subplots(1, n_dists, figsize=figsize, sharey=True)
    if n_dists == 1:
        axes = [axes]

    data = np.asarray(data, dtype=np.float64).flatten()

    colors = plt.cm.tab10.colors  # type: ignore
    for i, (spec, ax) in enumerate(zip(distributions, axes)):
        qq_plot(
            data,
            dist=spec.get("dist", stats.norm),
            dist_params=spec.get("params", {}),
            n_quantiles=n_quantiles,
            ax=ax,
            label=spec.get("label", f"Distribution {i+1}"),
            color=colors[i % len(colors)],
            show_confidence=True,
        )
        ax.set_title(spec.get("label", f"Distribution {i+1}"), fontsize=11)

    fig.suptitle("QQ Plot Comparison Across Distributions", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig
