"""
modeling_pipeline.py
--------------------
Standard workflow for fitting and comparing statistical models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.distributions.fitter import compare_distributions
from src.tails.tail_index import estimate_tail_index
from src.extreme_value.pot import extract_exceedances
from src.extreme_value.gpd import fit_gpd


def run_modeling_pipeline(
    data: NDArray[np.float64],
    dist_names: list[str] | None = None,
    tail_index_k: int | None = None,
    pot_threshold: float | None = None,
) -> dict[str, Any]:
    """Execute a complete distribution modeling pipeline.

    1. Fits and compares parametric distributions (Normal, Student-t, Stable).
    2. Estimates the tail index (Hill estimator).
    3. Fits a Generalized Pareto Distribution (GPD) via POT.

    Args:
        data: 1-D array of observations.
        dist_names: Distributions to compare. Defaults to all registered.
        tail_index_k: Number of extremes for Hill estimator. Defaults to sqrt(n).
        pot_threshold: Threshold for GPD fitting. Defaults to 95th percentile.

    Returns:
        Dictionary containing the results of each modeling step.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    n = len(data)

    # 1. Distribution Comparison
    print("Fitting and comparing distributions...")
    comparison_df = compare_distributions(data, dist_names)

    # 2. Tail Index Estimation
    if tail_index_k is None:
        tail_index_k = int(np.sqrt(n))
    
    print(f"Estimating tail index (k={tail_index_k})...")
    try:
        alpha_hat = estimate_tail_index(data, k=tail_index_k)
    except Exception as e:
        alpha_hat = float('nan')
        print(f"Warning: Tail index estimation failed: {e}")

    # 3. POT / GPD Fitting
    if pot_threshold is None:
        pot_threshold = float(np.quantile(data, 0.95))
    
    print(f"Fitting GPD at threshold={pot_threshold:.4f}...")
    try:
        exceedances = extract_exceedances(data, pot_threshold)
        gpd_results = fit_gpd(exceedances)
    except Exception as e:
        gpd_results = {}
        print(f"Warning: GPD fitting failed: {e}")

    return {
        "distribution_comparison": comparison_df,
        "tail_index": {
            "alpha": alpha_hat,
            "xi": 1.0 / alpha_hat if not np.isnan(alpha_hat) else float('nan'),
            "k": tail_index_k,
        },
        "gpd": gpd_results,
    }
