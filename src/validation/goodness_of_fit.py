"""
goodness_of_fit.py
------------------
Goodness-of-fit tests for parametric distribution fitting.

Tests assess whether fitted distributions are statistically consistent
with the observed data. Key tests implemented:
    - Kolmogorov-Smirnov (KS) test
    - Anderson-Darling (AD) test
    - Cramér-von Mises (CvM) test
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from typing import Callable


def kolmogorov_smirnov_test(
    data: NDArray[np.float64],
    cdf: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> dict[str, float]:
    """Perform the Kolmogorov-Smirnov goodness-of-fit test.

    Tests H0: data follows the specified distribution (given by cdf).
    The KS statistic is D = sup_x |F_n(x) - F(x)|, where F_n is the
    empirical CDF and F is the theoretical CDF.

    Args:
        data: 1-D array of observed values.
        cdf: Callable that takes an array and returns theoretical CDF values.
            Example: ``lambda x: scipy.stats.norm.cdf(x, loc=0, scale=1)``.

    Returns:
        Dictionary with:
            - ``statistic``: KS test statistic D.
            - ``p_value``: p-value for the test.
            - ``reject_h0_5pct``: True if H0 is rejected at the 5% level.

    Raises:
        ValueError: If data has fewer than 5 observations.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    if len(data) < 5:
        raise ValueError(f"KS test requires at least 5 observations, got {len(data)}.")

    stat, p_value = stats.kstest(data, cdf)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "reject_h0_5pct": bool(p_value < 0.05),
    }


def anderson_darling_test(
    data: NDArray[np.float64],
    dist_name: str = "norm",
) -> dict[str, float | list]:
    """Perform the Anderson-Darling goodness-of-fit test.

    The AD test is more sensitive than KS to differences in the tails,
    making it particularly useful for heavy-tail distribution assessment.

    Args:
        data: 1-D array of observed values.
        dist_name: Name of the distribution to test against. Supported by
            scipy: ``"norm"``, ``"expon"``, ``"logistic"``, ``"gumbel"``,
            ``"extreme1"``, ``"gumbel_l"``, ``"gumbel_r"``, ``"weibull_min"``.

    Returns:
        Dictionary with:
            - ``statistic``: AD test statistic.
            - ``critical_values``: Critical values at significance levels.
            - ``significance_levels``: Corresponding significance levels (e.g., [15, 10, 5, 2.5, 1]).
            - ``reject_h0_5pct``: True if statistic exceeds critical value at 5% level.

    Raises:
        ValueError: If data has fewer than 7 observations.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    if len(data) < 7:
        raise ValueError(f"Anderson-Darling test requires at least 7 observations, got {len(data)}.")

    result = stats.anderson(data, dist=dist_name)
    stat = float(result.statistic)
    critical_values = [float(c) for c in result.critical_values]
    significance_levels = [float(s) for s in result.significance_level]

    # Find critical value at ~5% significance level
    # scipy returns levels like [15, 10, 5, 2.5, 1] for norm
    reject_5pct = False
    for cv, sl in zip(critical_values, significance_levels):
        if abs(sl - 5.0) < 0.1:
            reject_5pct = stat > cv
            break

    return {
        "statistic": stat,
        "critical_values": critical_values,
        "significance_levels": significance_levels,
        "reject_h0_5pct": reject_5pct,
    }


def cramer_von_mises_test(
    data: NDArray[np.float64],
    cdf: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> dict[str, float]:
    """Perform the Cramér-von Mises goodness-of-fit test.

    The CvM statistic integrates the squared difference between empirical
    and theoretical CDFs:
        W^2 = sum_{i=1}^{n} [ F(x_(i)) - (2i-1)/(2n) ]^2 + 1/(12n)

    Args:
        data: 1-D array of observed values.
        cdf: Callable that takes an array and returns theoretical CDF values.

    Returns:
        Dictionary with:
            - ``statistic``: CvM test statistic W^2.
            - ``p_value``: Approximate p-value (from scipy).
            - ``reject_h0_5pct``: True if H0 is rejected at the 5% level.

    Raises:
        ValueError: If data has fewer than 5 observations.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    n = len(data)
    if n < 5:
        raise ValueError(f"CvM test requires at least 5 observations, got {n}.")

    sorted_data = np.sort(data)
    theoretical_probs = cdf(sorted_data)
    i_vals = np.arange(1, n + 1)
    empirical_probs = (2 * i_vals - 1) / (2 * n)

    w2 = float(np.sum((theoretical_probs - empirical_probs) ** 2) + 1.0 / (12 * n))

    # Approximate p-value using Monte Carlo simulation would be ideal,
    # but we use a known approximation for the asymptotic distribution.
    # p-value from scipy if available:
    try:
        p_val = float(stats.cramervonmises(sorted_data, cdf).pvalue)
    except Exception:
        # Fallback: rough approximation
        p_val = float(np.exp(-w2 * 6))
        p_val = float(np.clip(p_val, 0.0, 1.0))

    return {
        "statistic": w2,
        "p_value": p_val,
        "reject_h0_5pct": bool(p_val < 0.05),
    }


def goodness_of_fit_summary(
    data: NDArray[np.float64],
    cdf: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    dist_name: str = "norm",
) -> dict[str, dict[str, float]]:
    """Run all goodness-of-fit tests and return a combined summary.

    Args:
        data: 1-D array of observed values.
        cdf: Callable returning theoretical CDF values.
        dist_name: Distribution name for the Anderson-Darling test.

    Returns:
        Dictionary with keys ``"ks"``, ``"ad"``, ``"cvm"`` each containing
        the results from the respective tests.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    return {
        "ks": kolmogorov_smirnov_test(data, cdf),
        "ad": anderson_darling_test(data, dist_name),
        "cvm": cramer_von_mises_test(data, cdf),
    }
