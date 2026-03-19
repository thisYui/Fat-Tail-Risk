"""
tails submodule
---------------
Tail behavior analysis and index estimation.
"""

from __future__ import annotations

from .tail_index import (
    estimate_tail_index,
    hill_estimator,
    hill_plot_data,
    pickands_estimator,
    moments_estimator,
)
from .tail_metrics import (
    empirical_quantile,
    tail_probability,
    excess_distribution,
    mean_excess,
    tail_quantile,
    tail_conditional_expectation,
    tail_statistics,
)
from .tail_plots import (
    log_log_survival_plot,
    tail_qq_plot,
    hill_plot,
)

__all__ = [
    "estimate_tail_index",
    "hill_estimator",
    "hill_plot_data",
    "pickands_estimator",
    "moments_estimator",
    "empirical_quantile",
    "tail_probability",
    "excess_distribution",
    "mean_excess",
    "tail_quantile",
    "tail_conditional_expectation",
    "tail_statistics",
    "log_log_survival_plot",
    "tail_qq_plot",
    "hill_plot",
]
