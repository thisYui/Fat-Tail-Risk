"""
validation_pipeline.py
----------------------
Standard workflow for validating fitted models.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from src.validation.goodness_of_fit import goodness_of_fit_summary
from src.validation.statistical_tests import jarque_bera_test
from src.evaluation.metrics import tail_quantile_error


def run_validation_pipeline(
    data: NDArray[np.float64],
    model_cdf: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    model_ppf: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    dist_name: str = "norm",
) -> dict[str, Any]:
    """Execute a complete validation pipeline for a fitted model.

    1. Runs normality tests (Jarque-Bera).
    2. Performs goodness-of-fit tests (KS, AD, CvM).
    3. Computes tail quantile errors.

    Args:
        data: 1-D observed data.
        model_cdf: CDF function of the fitted model.
        model_ppf: Quantile (inverse CDF) function of the fitted model.
        dist_name: Name of distribution for AD test.

    Returns:
        Dictionary of validation results.
    """
    data = np.asarray(data, dtype=np.float64).flatten()

    # 1. Normality Check
    print("Running normality tests...")
    try:
        normality_results = jarque_bera_test(data)
    except Exception as e:
        normality_results = {"error": str(e)}

    # 2. Goodness of Fit
    print("Running goodness-of-fit tests...")
    try:
        gof_results = goodness_of_fit_summary(data, model_cdf, dist_name=dist_name)
    except Exception as e:
        gof_results = {"error": str(e)}

    # 3. Tail Errors
    print("Computing tail quantile errors...")
    try:
        tail_errors = tail_quantile_error(data, model_ppf)
    except Exception as e:
        tail_errors = {"error": str(e)}

    return {
        "normality": normality_results,
        "goodness_of_fit": gof_results,
        "tail_errors": tail_errors,
    }
