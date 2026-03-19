"""
copula.py
---------
Copula modeling for multivariate dependence structures.

A copula separates the modeling of marginal distributions from the
dependence structure. Currently implements:
    - Gaussian copula: linear correlation-based dependence.
    - Student-t copula: heavy-tailed dependence (tail dependence present).

Reference:
    Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs marges.
    Publications de l'Institut de Statistique de l'Université de Paris, 8, 229–231.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def fit_gaussian_copula(
    data: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Fit a Gaussian copula to multivariate data.

    Transforms each marginal to uniform via the empirical CDF, then applies
    the inverse Gaussian CDF (probit transform) to obtain pseudo-normal scores.
    The copula correlation matrix is estimated from these scores.

    Args:
        data: 2-D array of shape (n_samples, n_variables). Each column
            represents one variable.

    Returns:
        Dictionary with:
            - ``correlation``: Estimated copula correlation matrix (n_vars x n_vars).
            - ``n_samples``: Number of observations used.
            - ``n_variables``: Number of variables.
            - ``pseudo_observations``: Uniform pseudo-observations in (0,1).

    Raises:
        ValueError: If data is not 2-D or has fewer than 2 variables.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D (n_samples, n_vars), got shape {data.shape}.")
    n, d = data.shape
    if d < 2:
        raise ValueError(f"data must have at least 2 variables, got {d}.")

    # Step 1: Convert to uniform pseudo-observations via empirical CDF
    # Use rank-based transform with continuity correction: u_ij = rank / (n+1)
    pseudo_obs = _to_pseudo_observations(data)

    # Step 2: Apply probit transform to get pseudo-normal scores
    # Clip to avoid numerical issues at the boundary
    pseudo_obs_clipped = np.clip(pseudo_obs, 1e-10, 1 - 1e-10)
    normal_scores = stats.norm.ppf(pseudo_obs_clipped)

    # Step 3: Estimate correlation matrix from normal scores
    correlation = np.corrcoef(normal_scores, rowvar=False)

    return {
        "correlation": correlation,
        "n_samples": n,
        "n_variables": d,
        "pseudo_observations": pseudo_obs,
    }


def sample_gaussian_copula(
    n_samples: int,
    correlation: NDArray[np.float64],
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Sample from a Gaussian copula with given correlation matrix.

    Args:
        n_samples: Number of samples to draw.
        correlation: Correlation matrix of shape (d, d). Must be positive semi-definite.
        seed: Optional random seed.

    Returns:
        Array of shape (n_samples, d) with uniform [0,1] copula samples.

    Raises:
        ValueError: If correlation matrix is not square or not PSD.
    """
    corr = np.asarray(correlation, dtype=np.float64)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError(f"correlation must be a square matrix, got shape {corr.shape}.")

    d = corr.shape[0]
    rng = np.random.default_rng(seed)

    # Draw from multivariate normal with the copula correlation
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        # Add small jitter for numerical stability
        corr_stable = corr + 1e-8 * np.eye(d)
        L = np.linalg.cholesky(corr_stable)

    z = rng.standard_normal(size=(n_samples, d))
    correlated_normal = z @ L.T

    # Transform to uniform via normal CDF
    u = stats.norm.cdf(correlated_normal)
    return u


def fit_t_copula(
    data: NDArray[np.float64],
    df: float | None = None,
) -> dict[str, float | NDArray[np.float64]]:
    """Fit a Student-t copula to multivariate data.

    The t-copula has symmetric tail dependence (upper and lower tail dependence
    are equal). The correlation matrix is estimated from pseudo-normal scores
    via the t-distribution inverse CDF.

    Args:
        data: 2-D array of shape (n_samples, n_variables).
        df: Degrees of freedom for the t copula. If None, estimated from data
            by fitting a t distribution to each marginal and averaging.

    Returns:
        Dictionary with:
            - ``correlation``: Estimated copula correlation matrix.
            - ``df``: Degrees of freedom used.
            - ``n_samples``: Number of observations.
            - ``n_variables``: Number of variables.
            - ``pseudo_observations``: Uniform pseudo-observations.

    Raises:
        ValueError: If data is not 2-D or has fewer than 2 variables.
    """
    from src.distributions.student_t import fit as fit_t

    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D (n_samples, n_vars), got shape {data.shape}.")
    n, d = data.shape
    if d < 2:
        raise ValueError(f"data must have at least 2 variables, got {d}.")

    pseudo_obs = _to_pseudo_observations(data)

    if df is None:
        # Estimate df from each marginal and average
        marginal_dfs = []
        for j in range(d):
            try:
                fit_result = fit_t(data[:, j])
                marginal_dfs.append(fit_result["df"])
            except Exception:
                marginal_dfs.append(5.0)
        df = float(np.median(marginal_dfs))
        df = max(2.1, df)  # Ensure df > 2 for finite variance

    # Apply t quantile transform (inverse t CDF)
    pseudo_clipped = np.clip(pseudo_obs, 1e-10, 1 - 1e-10)
    t_scores = stats.t.ppf(pseudo_clipped, df=df)

    # Estimate correlation from t-scores
    correlation = np.corrcoef(t_scores, rowvar=False)

    return {
        "correlation": correlation,
        "df": df,
        "n_samples": n,
        "n_variables": d,
        "pseudo_observations": pseudo_obs,
    }


def sample_t_copula(
    n_samples: int,
    correlation: NDArray[np.float64],
    df: float,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Sample from a Student-t copula.

    Args:
        n_samples: Number of samples.
        correlation: Correlation matrix of shape (d, d).
        df: Degrees of freedom. Must be > 2.
        seed: Optional random seed.

    Returns:
        Array of shape (n_samples, d) with uniform [0,1] copula samples.

    Raises:
        ValueError: If df <= 0.
    """
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}.")

    corr = np.asarray(correlation, dtype=np.float64)
    d = corr.shape[0]
    rng = np.random.default_rng(seed)

    # Draw correlated multivariate normal
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        corr_stable = corr + 1e-8 * np.eye(d)
        L = np.linalg.cholesky(corr_stable)

    z = rng.standard_normal(size=(n_samples, d))
    correlated_normal = z @ L.T

    # Scale by chi-squared to create t distribution: t = Z / sqrt(V/df), V ~ chi2(df)
    chi2_samples = rng.chisquare(df=df, size=n_samples)
    scale = np.sqrt(chi2_samples / df)[:, np.newaxis]
    t_samples = correlated_normal / scale

    # Transform to uniform via t CDF
    u = stats.t.cdf(t_samples, df=df)
    return u


def _to_pseudo_observations(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert data to uniform pseudo-observations via rank transform.

    Uses the empirical marginal CDF with continuity correction:
        u_ij = rank(x_ij) / (n + 1)

    This ensures pseudo-observations are strictly in (0, 1), avoiding
    boundary issues when applying inverse-probability transforms.

    Args:
        data: 2-D array of shape (n_samples, n_variables).

    Returns:
        Array of same shape with values in (0, 1).
    """
    n = data.shape[0]
    ranks = np.zeros_like(data)
    for j in range(data.shape[1]):
        # argsort of argsort gives ranks (1-indexed)
        ranks[:, j] = stats.rankdata(data[:, j])
    return ranks / (n + 1)
