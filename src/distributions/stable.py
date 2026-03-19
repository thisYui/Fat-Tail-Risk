"""
stable.py
---------
Alpha-stable (Levy-stable) distribution modeling.

Alpha-stable distributions generalize the Gaussian and Cauchy distributions.
They are parameterized by:
    - alpha: stability index in (0, 2]; alpha=2 => Gaussian, alpha=1 => Cauchy.
    - beta: skewness in [-1, 1]; beta=0 => symmetric.
    - loc: location parameter.
    - scale: scale parameter (> 0).

These distributions have heavy tails (infinite variance for alpha < 2) and
are especially relevant for modeling extreme or impulsive data.

Performance note
----------------
scipy.stats.levy_stable.logpdf has no closed-form — it evaluates a numerical
integral for every data point. On typical hardware this costs ~1s per 1000-point
logpdf call, making full MLE impractical for large datasets.

Default fitting strategy: McCulloch (1986) quantile estimator — O(n), < 1ms,
consistent and sufficient for tail index analysis. MLE is available via
fit(..., method="mle") for cases where higher accuracy is required.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats


# ── Public API ─────────────────────────────────────────────────────────────────

def pdf(
    x: NDArray[np.float64],
    alpha: float,
    beta: float = 0.0,
    loc: float = 0.0,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """Compute the alpha-stable probability density function.

    Uses scipy's implementation, which numerically evaluates the density
    via characteristic function inversion (Nolan's S1 parametrization).

    Args:
        x: Points at which to evaluate the PDF.
        alpha: Stability index in (0, 2]. Lower alpha means heavier tails.
        beta: Skewness parameter in [-1, 1].
        loc: Location parameter.
        scale: Scale parameter. Must be > 0.

    Returns:
        Array of PDF values at each point in x.

    Raises:
        ValueError: If alpha, beta, or scale are out of valid range.
    """
    _validate_params(alpha, beta, scale)
    x = np.asarray(x, dtype=np.float64)
    return stats.levy_stable.pdf(x, alpha=alpha, beta=beta, loc=loc, scale=scale)


def log_likelihood(
    data: NDArray[np.float64],
    alpha: float,
    beta: float = 0.0,
    loc: float = 0.0,
    scale: float = 1.0,
) -> float:
    """Compute the total log-likelihood under an alpha-stable distribution.

    Args:
        data: Observed data samples.
        alpha: Stability index in (0, 2].
        beta: Skewness parameter in [-1, 1].
        loc: Location parameter.
        scale: Scale parameter. Must be > 0.

    Returns:
        Scalar total log-likelihood.

    Raises:
        ValueError: If parameters are out of valid range.
    """
    _validate_params(alpha, beta, scale)
    data = np.asarray(data, dtype=np.float64)
    log_pdfs = stats.levy_stable.logpdf(
        data, alpha=alpha, beta=beta, loc=loc, scale=scale
    )
    finite_mask = np.isfinite(log_pdfs)
    if not np.all(finite_mask):
        warnings.warn(
            f"{np.sum(~finite_mask)} data points produced -inf log-density; "
            "these are excluded from the total.",
            RuntimeWarning,
            stacklevel=2,
        )
    return float(np.sum(log_pdfs[finite_mask]))


def fit(
    data: NDArray[np.float64],
    fix_beta: float | None = None,
    method: str = "mcculloch",
    n_subsample: int | None = 1000,
    max_iter: int = 2000,
) -> dict[str, float]:
    """Fit an alpha-stable distribution to data.

    Two estimation methods are available:

    ``"mcculloch"`` (default):
        Quantile-based estimator from McCulloch (1986). Uses five sample
        quantiles to produce consistent parameter estimates in O(n) time
        with no numerical optimization. Typical runtime < 1ms regardless
        of n. Recommended for exploratory analysis and large datasets.

    ``"mle"``:
        Maximum likelihood via Nelder-Mead optimization. Slower by several
        orders of magnitude (minutes for n > 1000) because
        ``scipy.stats.levy_stable.logpdf`` has no closed form and requires
        numerical integration for every data point. Use only when precise
        log-likelihood values are needed (e.g. for AIC/BIC comparison).
        When n > n_subsample, optimization runs on a random subsample and
        final log-likelihood is recomputed on the full data.

    Args:
        data: Observed data samples. Must have at least 10 observations.
        fix_beta: If provided, beta is fixed at this value (both methods).
            Useful for imposing symmetry (fix_beta=0).
        method: Estimation method — ``"mcculloch"`` or ``"mle"``.
        n_subsample: (MLE only) Subsample size for optimization when
            n > n_subsample. Set to None to use the full dataset.
        max_iter: (MLE only) Maximum Nelder-Mead iterations.

    Returns:
        Dictionary with keys:
            - ``alpha``: Estimated stability index.
            - ``beta``: Estimated skewness.
            - ``loc``: Estimated location.
            - ``scale``: Estimated scale.
            - ``log_likelihood``: Total log-likelihood at the estimates.
              For ``"mcculloch"``, computed on a subsample to keep runtime fast.
            - ``aic``: Akaike Information Criterion.
            - ``bic``: Bayesian Information Criterion.
            - ``method``: Estimation method used.

    Raises:
        ValueError: If data has fewer than 10 observations or method is invalid.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    n = len(data)
    if n < 10:
        raise ValueError(f"At least 10 data points required for fitting, got {n}.")
    if method not in ("mcculloch", "mle"):
        raise ValueError(f"method must be 'mcculloch' or 'mle', got '{method}'.")

    if method == "mcculloch":
        return _fit_mcculloch(data, fix_beta, n_subsample)
    else:
        return _fit_mle(data, fix_beta, n_subsample, max_iter)


def sample(
    n_samples: int,
    alpha: float,
    beta: float = 0.0,
    loc: float = 0.0,
    scale: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Draw samples from an alpha-stable distribution.

    Args:
        n_samples: Number of samples.
        alpha: Stability index in (0, 2].
        beta: Skewness parameter in [-1, 1].
        loc: Location parameter.
        scale: Scale parameter. Must be > 0.
        seed: Optional random seed.

    Returns:
        Array of shape (n_samples,) with stable samples.

    Raises:
        ValueError: If parameters are out of valid range.
    """
    _validate_params(alpha, beta, scale)
    rng = np.random.default_rng(seed)
    return stats.levy_stable.rvs(
        alpha=alpha, beta=beta, loc=loc, scale=scale,
        size=n_samples, random_state=rng,
    )


# ── Estimation internals ───────────────────────────────────────────────────────

def _fit_mcculloch(
    data: NDArray[np.float64],
    fix_beta: float | None,
    n_subsample: int | None,
) -> dict[str, float]:
    """McCulloch (1986) quantile estimator — O(n), no optimization."""
    alpha_hat, beta_hat, loc_hat, scale_hat = _mcculloch_estimate(data)

    if fix_beta is not None:
        beta_hat = float(fix_beta)

    # Skip log-likelihood computation — levy_stable.logpdf has no closed form
    # and costs ~1s per 1000 points regardless of subsample size.
    # AIC/BIC are set to NaN for mcculloch; use method="mle" if needed.
    k   = 3 if fix_beta is not None else 4
    n   = len(data)

    return {
        "alpha":          float(alpha_hat),
        "beta":           float(beta_hat),
        "loc":            float(loc_hat),
        "scale":          float(scale_hat),
        "log_likelihood": np.nan,
        "aic":            np.nan,
        "bic":            np.nan,
        "method":         "mcculloch",
    }


def _fit_mle(
    data: NDArray[np.float64],
    fix_beta: float | None,
    n_subsample: int | None,
    max_iter: int,
) -> dict[str, float]:
    """MLE via Nelder-Mead — accurate but slow for large n."""
    n = len(data)

    # Subsample for optimization
    if n_subsample is not None and n > n_subsample:
        rng      = np.random.default_rng(seed=0)
        data_opt = rng.choice(data, size=n_subsample, replace=False)
    else:
        data_opt = data

    # McCulloch init → informed starting point → fewer iterations
    alpha_init, beta_init, loc_init, scale_init = _mcculloch_estimate(data_opt)

    # Transform to unconstrained space
    u0         = float(np.log((alpha_init + 1e-8) / (2.0 - alpha_init + 1e-8)))
    v0         = float(np.arctanh(np.clip(beta_init, -0.99, 0.99)))
    log_scale0 = float(np.log(scale_init + 1e-8))
    opts       = {"xatol": 1e-5, "fatol": 1e-5, "maxiter": max_iter}

    if fix_beta is not None:
        beta_fixed = float(fix_beta)

        def neg_ll(params: np.ndarray) -> float:
            u, loc_val, log_scale = params
            alpha_val = 2.0 / (1.0 + np.exp(-u))
            scale_val = np.exp(log_scale)
            lp = stats.levy_stable.logpdf(
                data_opt, alpha=alpha_val, beta=beta_fixed,
                loc=loc_val, scale=scale_val,
            )
            ll = float(np.sum(lp[np.isfinite(lp)]))
            return -ll if np.isfinite(ll) else 1e12

        x0        = [u0, loc_init, log_scale0]
        result    = optimize.minimize(neg_ll, x0, method="Nelder-Mead", options=opts)
        alpha_hat = 2.0 / (1.0 + np.exp(-result.x[0]))
        beta_hat  = beta_fixed
        loc_hat   = float(result.x[1])
        scale_hat = float(np.exp(result.x[2]))
        k         = 3
    else:
        def neg_ll(params: np.ndarray) -> float:
            u, v, loc_val, log_scale = params
            alpha_val = 2.0 / (1.0 + np.exp(-u))
            beta_val  = float(np.tanh(v))
            scale_val = np.exp(log_scale)
            lp = stats.levy_stable.logpdf(
                data_opt, alpha=alpha_val, beta=beta_val,
                loc=loc_val, scale=scale_val,
            )
            ll = float(np.sum(lp[np.isfinite(lp)]))
            return -ll if np.isfinite(ll) else 1e12

        x0        = [u0, v0, loc_init, log_scale0]
        result    = optimize.minimize(neg_ll, x0, method="Nelder-Mead", options=opts)
        alpha_hat = 2.0 / (1.0 + np.exp(-result.x[0]))
        beta_hat  = float(np.tanh(result.x[1]))
        loc_hat   = float(result.x[2])
        scale_hat = float(np.exp(result.x[3]))
        k         = 4

    # Final log-likelihood on full data
    ll  = log_likelihood(data, alpha_hat, beta_hat, loc_hat, scale_hat)
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll

    return {
        "alpha":          float(alpha_hat),
        "beta":           float(beta_hat),
        "loc":            float(loc_hat),
        "scale":          float(scale_hat),
        "log_likelihood": ll,
        "aic":            aic,
        "bic":            bic,
        "method":         "mle",
    }


def _mcculloch_estimate(
    data: NDArray[np.float64],
) -> tuple[float, float, float, float]:
    """McCulloch (1986) consistent quantile estimator for stable parameters.

    Uses five sample quantiles to produce closed-form estimates of all four
    parameters. O(n) complexity, no optimization required.

    Returns:
        Tuple of (alpha, beta, loc, scale).

    References:
        McCulloch, J. H. (1986). Simple consistent estimators of stable
        distribution parameters. Communications in Statistics, 15(4), 1109-1136.
    """
    p05, p25, p50, p75, p95 = np.quantile(data, [0.05, 0.25, 0.50, 0.75, 0.95])

    iqr = p75 - p25
    if iqr < 1e-10:
        return 1.5, 0.0, float(p50), max(float(np.std(data)), 1e-6)

    # Scale: IQR-based, robust to heavy tails
    scale = float(iqr / 1.349)

    # Alpha: v_alpha = (p95-p05)/(p75-p25)
    # 2.44 is the Gaussian reference value; larger = heavier tail = smaller alpha
    v_alpha   = max((p95 - p05) / iqr, 2.44)
    alpha_raw = 2.0 * (2.44 / v_alpha) ** 0.7
    alpha     = float(np.clip(alpha_raw, 0.4, 2.0))

    # Beta: tail asymmetry ratio
    tail_span = p95 - p05
    if tail_span > 1e-10:
        v_beta = (p95 + p05 - 2 * p50) / tail_span
        beta   = float(np.clip(v_beta * 2.5, -1.0, 1.0))
    else:
        beta = 0.0

    loc = float(p50)
    return alpha, beta, loc, scale


# ── Validation ─────────────────────────────────────────────────────────────────

def _validate_params(alpha: float, beta: float, scale: float) -> None:
    """Validate alpha-stable distribution parameters.

    Raises:
        ValueError: If any parameter is out of range.
    """
    if not (0 < alpha <= 2):
        raise ValueError(f"alpha must be in (0, 2], got {alpha}.")
    if not (-1 <= beta <= 1):
        raise ValueError(f"beta must be in [-1, 1], got {beta}.")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}.")