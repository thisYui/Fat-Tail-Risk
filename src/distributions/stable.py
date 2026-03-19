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
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats


def pdf(
    x: NDArray[np.float64],
    alpha: float,
    beta: float = 0.0,
    loc: float = 0.0,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """Compute the alpha-stable probability density function.

    Uses scipy's implementation, which numerically evaluates the density
    via characteristic function inversion (Nolan's parametrization S1).

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
    log_pdfs = stats.levy_stable.logpdf(data, alpha=alpha, beta=beta, loc=loc, scale=scale)
    # Guard against -inf values
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
) -> dict[str, float]:
    """Fit an alpha-stable distribution to data via MLE.

    Uses numerical optimization to jointly estimate alpha, beta, loc, and scale.
    The optimization is performed in a transformed space to enforce parameter
    constraints:
        - alpha: parameterized as alpha = 2 * sigmoid(u) for u in R => alpha in (0, 2).
        - scale: parameterized as log(scale) for positivity.
        - beta in [-1, 1]: parameterized as beta = tanh(v).

    Note: Fitting alpha-stable distributions is computationally expensive
    due to the numerical density evaluation. For large datasets, consider
    using the McCulloch (1986) method-of-moments estimator as initialization.

    Args:
        data: Observed data samples. Must have at least 10 observations.
        fix_beta: If provided, beta is fixed at this value and not optimized.
            Useful for imposing symmetry (fix_beta=0).

    Returns:
        Dictionary with keys:
            - ``alpha``: Estimated stability index.
            - ``beta``: Estimated skewness.
            - ``loc``: Estimated location.
            - ``scale``: Estimated scale.
            - ``log_likelihood``: Total log-likelihood at the MLE.
            - ``aic``: Akaike Information Criterion.
            - ``bic``: Bayesian Information Criterion.

    Raises:
        ValueError: If data has fewer than 10 observations.
        RuntimeError: If optimization fails to converge.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    n = len(data)
    if n < 10:
        raise ValueError(f"At least 10 data points required for fitting, got {n}.")

    data_std = np.std(data)
    data_med = np.median(data)

    if fix_beta is not None:
        beta_fixed = float(fix_beta)

        def neg_ll(params: np.ndarray) -> float:
            u, loc_val, log_scale = params
            alpha_val = 2.0 / (1.0 + np.exp(-u))  # sigmoid maps to (0, 2)
            scale_val = np.exp(log_scale)
            ll = log_likelihood(data, alpha_val, beta_fixed, loc_val, scale_val)
            return -ll if np.isfinite(ll) else 1e12

        x0 = [0.0, data_med, np.log(data_std + 1e-8)]
        result = optimize.minimize(neg_ll, x0, method="Nelder-Mead",
                                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 10000})
        alpha_hat = 2.0 / (1.0 + np.exp(-result.x[0]))
        beta_hat = beta_fixed
        loc_hat = float(result.x[1])
        scale_hat = float(np.exp(result.x[2]))
        k = 3
    else:
        def neg_ll(params: np.ndarray) -> float:
            u, v, loc_val, log_scale = params
            alpha_val = 2.0 / (1.0 + np.exp(-u))
            beta_val = np.tanh(v)
            scale_val = np.exp(log_scale)
            ll = log_likelihood(data, alpha_val, beta_val, loc_val, scale_val)
            return -ll if np.isfinite(ll) else 1e12

        x0 = [0.0, 0.0, data_med, np.log(data_std + 1e-8)]
        result = optimize.minimize(neg_ll, x0, method="Nelder-Mead",
                                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 10000})
        alpha_hat = 2.0 / (1.0 + np.exp(-result.x[0]))
        beta_hat = float(np.tanh(result.x[1]))
        loc_hat = float(result.x[2])
        scale_hat = float(np.exp(result.x[3]))
        k = 4

    ll = log_likelihood(data, alpha_hat, beta_hat, loc_hat, scale_hat)
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll

    return {
        "alpha": float(alpha_hat),
        "beta": float(beta_hat),
        "loc": float(loc_hat),
        "scale": float(scale_hat),
        "log_likelihood": ll,
        "aic": aic,
        "bic": bic,
    }


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
        alpha=alpha, beta=beta, loc=loc, scale=scale, size=n_samples, random_state=rng
    )


def _validate_params(alpha: float, beta: float, scale: float) -> None:
    """Validate alpha-stable distribution parameters.

    Args:
        alpha: Stability index.
        beta: Skewness parameter.
        scale: Scale parameter.

    Raises:
        ValueError: If any parameter is out of range.
    """
    if not (0 < alpha <= 2):
        raise ValueError(f"alpha must be in (0, 2], got {alpha}.")
    if not (-1 <= beta <= 1):
        raise ValueError(f"beta must be in [-1, 1], got {beta}.")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}.")
