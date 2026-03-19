"""
Fat-Tail Risk Package
--------------------
A research-grade library for statistical modeling of heavy-tailed
distributions and extreme values.
"""

from __future__ import annotations

from . import (
    data,
    dependence,
    distributions,
    evaluation,
    extreme_value,
    pipelines,
    simulation,
    tails,
    utils,
    validation,
)

__version__ = "0.1.0"

__all__ = [
    "data",
    "dependence",
    "distributions",
    "evaluation",
    "extreme_value",
    "pipelines",
    "simulation",
    "tails",
    "utils",
    "validation",
]
