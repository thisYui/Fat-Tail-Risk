"""
data submodule
--------------
Synthetic data generation and loading utilities.
"""

from __future__ import annotations

from .generators import (
    gaussian,
    student_t,
    pareto,
    lognormal,
    skewed_student_t,
    generalized_pareto,
    mixed_distribution,
)
from .loaders import (
    load_csv,
    load_parquet,
    load_numpy,
    validate_data,
    load_dataframe,
)

__all__ = [
    "gaussian",
    "student_t",
    "pareto",
    "lognormal",
    "skewed_student_t",
    "generalized_pareto",
    "mixed_distribution",
    "load_csv",
    "load_parquet",
    "load_numpy",
    "validate_data",
    "load_dataframe",
]
