"""
dependence submodule
--------------------
Multivariate dependence modeling and copulas.
"""

from __future__ import annotations

from .copula import (
    fit_gaussian_copula,
    sample_gaussian_copula,
    fit_t_copula,
    sample_t_copula,
)
from .tail_dependence import (
    upper_tail_dependence,
    lower_tail_dependence,
    tail_dependence_profile,
    kendall_tau,
    spearman_rho,
)

__all__ = [
    "fit_gaussian_copula",
    "sample_gaussian_copula",
    "fit_t_copula",
    "sample_t_copula",
    "upper_tail_dependence",
    "lower_tail_dependence",
    "tail_dependence_profile",
    "kendall_tau",
    "spearman_rho",
]
