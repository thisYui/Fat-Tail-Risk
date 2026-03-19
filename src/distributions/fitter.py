"""
fitter.py
---------
Unified distribution fitting and model comparison interface.

Provides a common API for fitting multiple parametric distributions to data
and comparing them using information criteria (AIC, BIC) and log-likelihood.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.distributions import normal, student_t, stable


# Registry mapping distribution names to their fit functions
_DISTRIBUTION_REGISTRY: dict[str, Any] = {
    "normal": normal.fit,
    "student_t": student_t.fit,
    "stable": stable.fit,
}


def fit_distribution(
    data: NDArray[np.float64],
    dist_name: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Fit a single named distribution to data using MLE.

    Args:
        data: Observed data samples.
        dist_name: Distribution to fit. Supported values:
            ``"normal"``, ``"student_t"``, ``"stable"``.
        **kwargs: Additional keyword arguments passed to the underlying fit function.

    Returns:
        Dictionary of fitted parameters plus goodness-of-fit metrics:
            - ``distribution``: Name of the fitted distribution.
            - Parameter estimates (keys depend on distribution).
            - ``log_likelihood``: Total log-likelihood at the MLE.
            - ``aic``: Akaike Information Criterion.
            - ``bic``: Bayesian Information Criterion.

    Raises:
        ValueError: If dist_name is not in the registry.
        RuntimeError: If the fitting procedure fails.
    """
    data = np.asarray(data, dtype=np.float64).flatten()

    if dist_name not in _DISTRIBUTION_REGISTRY:
        available = list(_DISTRIBUTION_REGISTRY.keys())
        raise ValueError(
            f"Unknown distribution '{dist_name}'. Available: {available}"
        )

    fit_fn = _DISTRIBUTION_REGISTRY[dist_name]
    result = fit_fn(data, **kwargs)
    result["distribution"] = dist_name
    return result


def compare_distributions(
    data: NDArray[np.float64],
    dist_names: list[str] | None = None,
    **kwargs: dict[str, Any],
) -> pd.DataFrame:
    """Fit multiple distributions to data and compare them by AIC/BIC.

    Args:
        data: Observed data samples.
        dist_names: List of distribution names to fit. Defaults to all
            registered distributions: ``["normal", "student_t", "stable"]``.
        **kwargs: Per-distribution keyword arguments. Keys are distribution
            names and values are dicts of kwargs, e.g.
            ``{"student_t": {"fix_loc": 0.0}}``.

    Returns:
        DataFrame sorted by AIC (ascending) with columns:
            - ``distribution``: Name of the distribution.
            - ``log_likelihood``: Log-likelihood at the MLE.
            - ``aic``: AIC value (lower is better).
            - ``bic``: BIC value.
            - Additional columns for distribution-specific parameters.

    Raises:
        ValueError: If any dist_name is not in the registry.
    """
    data = np.asarray(data, dtype=np.float64).flatten()

    if dist_names is None:
        dist_names = list(_DISTRIBUTION_REGISTRY.keys())

    results = []
    for name in dist_names:
        dist_kwargs = kwargs.get(name, {})
        try:
            result = fit_distribution(data, name, **dist_kwargs)
            results.append(result)
        except Exception as exc:
            results.append({
                "distribution": name,
                "log_likelihood": np.nan,
                "aic": np.nan,
                "bic": np.nan,
                "error": str(exc),
            })

    df = pd.DataFrame(results)

    # Move 'distribution' to the first column
    cols = ["distribution"] + [c for c in df.columns if c != "distribution"]
    df = df[cols]

    # Sort by AIC, putting NaN at the end
    df = df.sort_values("aic", na_position="last").reset_index(drop=True)
    return df


def best_distribution(
    data: NDArray[np.float64],
    dist_names: list[str] | None = None,
    criterion: str = "aic",
) -> dict[str, Any]:
    """Select the best-fitting distribution based on an information criterion.

    Args:
        data: Observed data samples.
        dist_names: Distribution names to consider. Defaults to all registered.
        criterion: Information criterion to optimize. Either ``"aic"`` or ``"bic"``.

    Returns:
        Dictionary of fitted parameters for the best distribution, including
        the ``distribution`` key and AIC/BIC metrics.

    Raises:
        ValueError: If criterion is not ``"aic"`` or ``"bic"``.
        RuntimeError: If no distribution was successfully fitted.
    """
    if criterion not in ("aic", "bic"):
        raise ValueError(f"criterion must be 'aic' or 'bic', got '{criterion}'.")

    comparison = compare_distributions(data, dist_names)
    valid = comparison.dropna(subset=[criterion])

    if valid.empty:
        raise RuntimeError("All distribution fits failed; no valid results to compare.")

    best_row = valid.loc[valid[criterion].idxmin()]
    return best_row.dropna().to_dict()


def list_distributions() -> list[str]:
    """Return a list of all registered distribution names.

    Returns:
        List of distribution name strings.
    """
    return list(_DISTRIBUTION_REGISTRY.keys())
