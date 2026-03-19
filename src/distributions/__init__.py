"""
distributions submodule
-----------------------
Parametric distribution modeling and MLE fitting.
"""

from __future__ import annotations

from .fitter import (
    fit_distribution,
    compare_distributions,
    best_distribution,
    list_distributions,
)

__all__ = [
    "fit_distribution",
    "compare_distributions",
    "best_distribution",
    "list_distributions",
]
