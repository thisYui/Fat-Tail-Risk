"""
plotting.py
-----------
Reusable plotting utilities for statistical visualization.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_histogram_with_density(
    data: NDArray[np.float64],
    pdf_fns: list[tuple[str, Callable[[NDArray[np.float64]], NDArray[np.float64]]]] | None = None,
    ax: plt.Axes | None = None,
    bins: int = 50,
    title: str = "Data Distribution with Fitted Densities",
) -> plt.Axes:
    """Plot histogram of data with optional overlaid PDF curves.

    Args:
        data: 1-D observed data.
        pdf_fns: List of (label, pdf_callable) for overlaid distribution curves.
        ax: Matplotlib axes.
        bins: Number of histogram bins.
        title: Plot title.

    Returns:
        Matplotlib Axes.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.hist(data, bins=bins, density=True, alpha=0.5, color="gray", label="Empirical")

    if pdf_fns:
        x_range = np.linspace(np.min(data), np.max(data), 500)
        for label, fn in pdf_fns:
            ax.plot(x_range, fn(x_range), label=label, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def set_publication_style() -> None:
    """Set matplotlib parameters for research-grade plots."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": (8, 5),
        "grid.alpha": 0.4,
    })
