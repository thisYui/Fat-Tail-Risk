"""
statistical_tests.py
--------------------
Statistical hypothesis tests for distribution model comparison.

Implements:
    - Likelihood Ratio Test (LRT): tests nested model fit improvement.
    - Vuong Test: test for non-nested model comparison.
    - Jarque-Bera normality test.
    - D'Agostino-Pearson normality test.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def likelihood_ratio_test(
    log_likelihood_null: float,
    log_likelihood_alt: float,
    df: int,
) -> dict[str, float]:
    """Perform a Likelihood Ratio Test (LRT) for nested models.

    H0: The null (simpler) model is adequate.
    H1: The alternative (more complex) model provides significantly better fit.

    LRT statistic: Lambda = -2 * (LL_null - LL_alt) ~ chi^2(df) under H0.

    Args:
        log_likelihood_null: Log-likelihood of the null (restricted) model.
        log_likelihood_alt: Log-likelihood of the alternative (full) model.
        df: Degrees of freedom = number of additional parameters in the
            alternative model.

    Returns:
        Dictionary with:
            - ``statistic``: LRT statistic (chi-squared value).
            - ``p_value``: p-value from chi-squared distribution.
            - ``df``: Degrees of freedom.
            - ``reject_h0_5pct``: True if H0 is rejected at 5% significance.

    Raises:
        ValueError: If df < 1 or if LL_alt < LL_null (alt should always be >= null).
    """
    if df < 1:
        raise ValueError(f"Degrees of freedom must be >= 1, got {df}.")

    lrt_stat = -2.0 * (log_likelihood_null - log_likelihood_alt)
    if lrt_stat < -1e-6:  # allow small numerical tolerance
        raise ValueError(
            f"LRT statistic is negative ({lrt_stat:.4f}). The alternative model "
            "must have log-likelihood >= the null model."
        )
    lrt_stat = max(0.0, lrt_stat)

    p_value = float(1.0 - stats.chi2.cdf(lrt_stat, df=df))

    return {
        "statistic": float(lrt_stat),
        "p_value": p_value,
        "df": df,
        "reject_h0_5pct": bool(p_value < 0.05),
    }


def vuong_test(
    log_likelihoods_1: NDArray[np.float64],
    log_likelihoods_2: NDArray[np.float64],
) -> dict[str, float]:
    """Perform the Vuong (1989) test for non-nested model comparison.

    Tests H0: model 1 and model 2 are equally close to the true distribution.
    H1: One model is strictly closer.

    The test statistic is approximately standard normal under H0.

    Args:
        log_likelihoods_1: Vector of per-observation log-likelihoods for model 1.
        log_likelihoods_2: Vector of per-observation log-likelihoods for model 2.

    Returns:
        Dictionary with:
            - ``statistic``: Vuong test statistic (z-score).
            - ``p_value``: Two-sided p-value.
            - ``preferred_model``: ``"model_1"``, ``"model_2"``, or ``"equal"``.

    Raises:
        ValueError: If arrays have different lengths or fewer than 10 observations.

    References:
        Vuong, Q. H. (1989). Likelihood Ratio Tests for Model Selection and
        Non-Nested Hypotheses. Econometrica, 57(2), 307-333.
    """
    ll1 = np.asarray(log_likelihoods_1, dtype=np.float64).flatten()
    ll2 = np.asarray(log_likelihoods_2, dtype=np.float64).flatten()

    if len(ll1) != len(ll2):
        raise ValueError(
            f"log_likelihoods arrays must have equal length, got {len(ll1)} and {len(ll2)}."
        )
    n = len(ll1)
    if n < 10:
        raise ValueError(f"At least 10 observations required for Vuong test, got {n}.")

    # Difference in log-likelihoods per observation
    m_i = ll1 - ll2
    m_bar = np.mean(m_i)
    sigma_m = np.std(m_i, ddof=1)

    if sigma_m < 1e-10:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "preferred_model": "equal",
        }

    z = float(np.sqrt(n) * m_bar / sigma_m)
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))

    if p_value < 0.05:
        preferred = "model_1" if z > 0 else "model_2"
    else:
        preferred = "equal"

    return {
        "statistic": z,
        "p_value": p_value,
        "preferred_model": preferred,
    }


def jarque_bera_test(
    data: NDArray[np.float64],
) -> dict[str, float]:
    """Perform the Jarque-Bera test for normality.

    Tests H0: data is normally distributed.
    H1: data has non-zero skewness or non-zero excess kurtosis.

    The JB statistic combines sample skewness and excess kurtosis.

    Args:
        data: 1-D array of observed values. Requires at least 20 observations.

    Returns:
        Dictionary with:
            - ``statistic``: JB test statistic.
            - ``p_value``: p-value from chi-squared(2) distribution.
            - ``skewness``: Sample skewness.
            - ``excess_kurtosis``: Sample excess kurtosis.
            - ``reject_h0_5pct``: True if H0 is rejected at 5%.

    Raises:
        ValueError: If fewer than 20 observations.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    n = len(data)
    if n < 20:
        raise ValueError(f"Jarque-Bera test requires >= 20 observations, got {n}.")

    skewness = float(stats.skew(data))
    excess_kurt = float(stats.kurtosis(data, fisher=True))  # excess (normal=0)
    jb_stat = float(n / 6.0 * (skewness**2 + excess_kurt**2 / 4.0))
    p_value = float(1.0 - stats.chi2.cdf(jb_stat, df=2))

    return {
        "statistic": jb_stat,
        "p_value": p_value,
        "skewness": skewness,
        "excess_kurtosis": excess_kurt,
        "reject_h0_5pct": bool(p_value < 0.05),
    }


def dagostino_pearson_test(
    data: NDArray[np.float64],
) -> dict[str, float]:
    """Perform the D'Agostino-Pearson omnibus normality test.

    Combines separate tests for skewness and kurtosis into a single
    chi-squared test.

    Args:
        data: 1-D array of observed values.

    Returns:
        Dictionary with:
            - ``statistic``: Test statistic (chi-squared).
            - ``p_value``: p-value.
            - ``reject_h0_5pct``: True if rejected at 5%.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    stat, p_value = stats.normaltest(data)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "reject_h0_5pct": bool(p_value < 0.05),
    }


def shapiro_wilk_test(
    data: NDArray[np.float64],
) -> dict[str, float]:
    """Perform the Shapiro-Wilk test for normality.

    Most powerful normality test for small-to-medium samples (n <= 5000).

    Args:
        data: 1-D array of observed values. Must have 3 to 5000 observations.

    Returns:
        Dictionary with:
            - ``statistic``: Shapiro-Wilk W statistic.
            - ``p_value``: p-value.
            - ``reject_h0_5pct``: True if rejected at 5%.

    Raises:
        ValueError: If sample size is outside [3, 5000].
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    n = len(data)
    if not (3 <= n <= 5000):
        raise ValueError(
            f"Shapiro-Wilk requires 3 <= n <= 5000 observations, got {n}."
        )
    stat, p_value = stats.shapiro(data)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "reject_h0_5pct": bool(p_value < 0.05),
    }
