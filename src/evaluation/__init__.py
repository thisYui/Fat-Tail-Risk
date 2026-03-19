"""
evaluation submodule
--------------------
Performance evaluation and uncertainty quantification.
"""

from __future__ import annotations

from .metrics import (
    quantile_error,
    tail_quantile_error,
    tail_probability_error,
    wasserstein_distance,
    kolmogorov_smirnov_distance,
)
from .uncertainty import (
    bootstrap_confidence_interval,
    bootstrap_parameter_cis,
    empirical_coverage,
)

__all__ = [
    "quantile_error",
    "tail_quantile_error",
    "tail_probability_error",
    "wasserstein_distance",
    "kolmogorov_smirnov_distance",
    "bootstrap_confidence_interval",
    "bootstrap_parameter_cis",
    "empirical_coverage",
]
