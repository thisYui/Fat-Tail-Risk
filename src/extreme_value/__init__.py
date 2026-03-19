"""
extreme_value submodule
-----------------------
Extreme Value Theory (EVT) and Peaks-Over-Threshold (POT) modeling.
"""

from __future__ import annotations

from .pot import (
    extract_exceedances,
    pot_summary,
    threshold_range_analysis,
)
from .gpd import (
    fit_gpd,
    gpd_quantile,
    gpd_tail_probability,
)
from .threshold_selection import (
    mean_excess_function,
    stability_plot_data,
    plot_mean_excess,
    plot_stability,
)

__all__ = [
    "extract_exceedances",
    "pot_summary",
    "threshold_range_analysis",
    "fit_gpd",
    "gpd_quantile",
    "gpd_tail_probability",
    "mean_excess_function",
    "stability_plot_data",
    "plot_mean_excess",
    "plot_stability",
]
