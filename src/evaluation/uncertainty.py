"""
uncertainty.py
--------------
Bootstrap and sampling-based uncertainty quantification.

Provides bootstrap confidence intervals for distribution parameters,
tail index estimates, and quantile estimates. Bootstrapping is a
non-parametric approach that makes no assumption about the sampling
distribution of the estimator.

Performance notes (BCa method):
    The BCa acceleration factor requires a jackknife pass over the data,
    which is O(n) fits and dominates runtime for large n. This implementation
    caps the jackknife at ``_JACK_MAX`` observations (subsampled without
    replacement) — the acceleration factor 'a' is a smooth functional of the
    data and is stable at n=200 for typical tail estimators. Bootstrap
    replicates are unaffected; only the jackknife subsample is reduced.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

# Maximum jackknife sample size for BCa acceleration factor.
# Full jackknife at n=5000 costs ~5000 model fits per CI call.
# Subsampling to 200 reduces this 25x with negligible accuracy loss.
_JACK_MAX = 200


def bootstrap_confidence_interval(
    data: NDArray[np.float64],
    statistic_fn: Callable[[NDArray[np.float64]], float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    method: str = "percentile",
    seed: int | None = None,
    bca_jack_max: int = _JACK_MAX,
) -> dict[str, float]:
    """Compute a bootstrap confidence interval for a scalar statistic.

    Args:
        data: 1-D array of observed values.
        statistic_fn: Function that takes a data array and returns a scalar.
            Example: ``lambda x: np.mean(x)`` or a tail index estimator.
        n_bootstrap: Number of bootstrap replications.
        confidence_level: Coverage level, e.g., 0.95 for 95% CI.
        method: CI construction method.
            - ``"percentile"``: Uses bootstrap percentiles directly.
            - ``"basic"``: Pivotal (reflection) method; more robust.
            - ``"bca"``: Bias-corrected and accelerated method (most accurate).
        seed: Optional random seed.
        bca_jack_max: Maximum number of observations used for the BCa
            jackknife pass. When ``len(data) > bca_jack_max``, a random
            subsample of size ``bca_jack_max`` is used to estimate the
            acceleration factor 'a'. Has no effect for other methods.
            Default: 200.

    Returns:
        Dictionary with:
            - ``estimate``: Point estimate on the original data.
            - ``ci_lower``: Lower bound of confidence interval.
            - ``ci_upper``: Upper bound of confidence interval.
            - ``se``: Bootstrap standard error.
            - ``bias``: Bootstrap bias estimate (bootstrap_mean - estimate).

    Raises:
        ValueError: If confidence_level not in (0, 1) or method is unknown.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    n = len(data)

    if not (0 < confidence_level < 1):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}.")
    if method not in ("percentile", "basic", "bca"):
        raise ValueError(f"method must be 'percentile', 'basic', or 'bca', got '{method}'.")

    estimate = float(statistic_fn(data))
    rng = np.random.default_rng(seed)

    # ── Bootstrap replicates ─────────────────────────────────────────────────
    bootstrap_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resample = rng.choice(data, size=n, replace=True)
        bootstrap_stats[i] = float(statistic_fn(resample))

    se   = float(np.std(bootstrap_stats, ddof=1))
    bias = float(np.mean(bootstrap_stats) - estimate)

    alpha = 1.0 - confidence_level
    lo    = alpha / 2
    hi    = 1.0 - lo

    # ── CI bounds ────────────────────────────────────────────────────────────
    if method == "percentile":
        ci_lower = float(np.nanpercentile(bootstrap_stats, lo * 100))
        ci_upper = float(np.nanpercentile(bootstrap_stats, hi * 100))

    elif method == "basic":
        ci_lower = float(2 * estimate - np.nanpercentile(bootstrap_stats, hi * 100))
        ci_upper = float(2 * estimate - np.nanpercentile(bootstrap_stats, lo * 100))

    elif method == "bca":
        # ── Bias-correction factor z0 ────────────────────────────────────────
        prop_below = float(np.mean(bootstrap_stats < estimate))

        if prop_below in (0.0, 1.0):
            # Degenerate bootstrap distribution: fall back to percentile
            ci_lower = float(np.nanpercentile(bootstrap_stats, lo * 100))
            ci_upper = float(np.nanpercentile(bootstrap_stats, hi * 100))

        else:
            from scipy.stats import norm

            z0 = float(norm.ppf(prop_below))

            # ── Acceleration factor 'a' via jackknife ────────────────────────
            # Subsample to bca_jack_max if n is large — 'a' is a smooth
            # functional and is stable well before n=200 observations.
            # Using boolean-mask indexing avoids np.delete() array copies.
            if n > bca_jack_max:
                jack_data = data[rng.choice(n, size=bca_jack_max, replace=False)]
                jack_n    = bca_jack_max
            else:
                jack_data = data
                jack_n    = n

            jackknife_stats = np.empty(jack_n)
            idx = np.arange(jack_n)
            for j in range(jack_n):
                jackknife_stats[j] = float(statistic_fn(jack_data[idx != j]))

            jack_mean = np.mean(jackknife_stats)
            diff      = jack_mean - jackknife_stats          # shape (jack_n,)
            num       = float(np.sum(diff ** 3))
            den       = float(6.0 * (np.sum(diff ** 2)) ** 1.5)
            a         = num / den if abs(den) > 1e-10 else 0.0

            # ── Adjusted percentile levels ───────────────────────────────────
            z_lo   = float(norm.ppf(lo))
            z_hi   = float(norm.ppf(hi))
            adj_lo = norm.cdf(z0 + (z0 + z_lo) / (1.0 - a * (z0 + z_lo))) * 100
            adj_hi = norm.cdf(z0 + (z0 + z_hi) / (1.0 - a * (z0 + z_hi))) * 100
            ci_lower = float(np.nanpercentile(bootstrap_stats, adj_lo))
            ci_upper = float(np.nanpercentile(bootstrap_stats, adj_hi))

    return {
        "estimate": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se":       se,
        "bias":     bias,
    }


def bootstrap_parameter_cis(
    data: NDArray[np.float64],
    fit_fn: Callable[[NDArray[np.float64]], dict[str, float]],
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> dict[str, dict[str, float]]:
    """Compute bootstrap confidence intervals for all parameters returned by a fit function.

    Args:
        data: 1-D observed data.
        fit_fn: Function that takes data and returns a dict of parameter estimates.
            Example: ``normal.fit`` returns ``{"mu": ..., "sigma": ..., ...}``.
        n_bootstrap: Number of bootstrap replications.
        confidence_level: Coverage level.
        seed: Optional random seed.

    Returns:
        Dictionary keyed by parameter name, each containing:
            - ``estimate``: Point estimate.
            - ``ci_lower``: Lower CI bound.
            - ``ci_upper``: Upper CI bound.
            - ``se``: Bootstrap standard error.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    n    = len(data)

    original_params = fit_fn(data)
    rng             = np.random.default_rng(seed)
    alpha           = 1.0 - confidence_level

    param_keys = [k for k, v in original_params.items() if isinstance(v, (int, float))]
    bootstrap_results: dict[str, list[float]] = {k: [] for k in param_keys}

    for _ in range(n_bootstrap):
        resample = rng.choice(data, size=n, replace=True)
        try:
            params = fit_fn(resample)
            for k in param_keys:
                if k in params:
                    bootstrap_results[k].append(float(params[k]))
        except Exception:
            continue

    ci_dict: dict[str, dict[str, float]] = {}
    for k in param_keys:
        vals = np.array(bootstrap_results[k])
        if len(vals) < 10:
            ci_dict[k] = {
                "estimate": float(original_params[k]),
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "se":       np.nan,
            }
        else:
            ci_dict[k] = {
                "estimate": float(original_params[k]),
                "ci_lower": float(np.nanpercentile(vals, alpha / 2 * 100)),
                "ci_upper": float(np.nanpercentile(vals, (1 - alpha / 2) * 100)),
                "se":       float(np.std(vals, ddof=1)),
            }

    return ci_dict


def empirical_coverage(
    data: NDArray[np.float64],
    lower_bound_fn: Callable[[NDArray[np.float64]], float],
    upper_bound_fn: Callable[[NDArray[np.float64]], float],
    n_bootstrap: int = 500,
    seed: int | None = None,
) -> float:
    """Estimate empirical coverage of a confidence interval via the double bootstrap.

    Args:
        data: 1-D observed data.
        lower_bound_fn: Callable returning the lower CI bound from data.
        upper_bound_fn: Callable returning the upper CI bound from data.
        n_bootstrap: Number of bootstrap replications.
        seed: Optional random seed.

    Returns:
        Empirical coverage fraction (proportion of bootstrap samples where the
        true estimate falls within the CI).
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    n    = len(data)

    rng         = np.random.default_rng(seed)
    true_lower  = lower_bound_fn(data)
    true_upper  = upper_bound_fn(data)

    n_covered = 0
    for _ in range(n_bootstrap):
        resample = rng.choice(data, size=n, replace=True)
        try:
            lo = lower_bound_fn(resample)
            hi = upper_bound_fn(resample)
            if true_lower <= hi and lo <= true_upper:
                n_covered += 1
        except Exception:
            continue

    return float(n_covered / n_bootstrap)