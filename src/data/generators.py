"""
generators.py
-------------
Synthetic dataset generation for statistical modeling.

Provides functions to generate samples from Gaussian, Student-t,
and other heavy-tailed distributions for simulation and testing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def gaussian(
    n_samples: int,
    mean: float = 0.0,
    std: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate samples from a Gaussian (Normal) distribution.

    Args:
        n_samples: Number of samples to generate.
        mean: Mean of the distribution.
        std: Standard deviation of the distribution.
        seed: Optional random seed for reproducibility.

    Returns:
        Array of shape (n_samples,) with Gaussian samples.

    Raises:
        ValueError: If n_samples <= 0 or std <= 0.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}.")
    if std <= 0:
        raise ValueError(f"std must be positive, got {std}.")

    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=std, size=n_samples)


def student_t(
    n_samples: int,
    df: float,
    loc: float = 0.0,
    scale: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate samples from a Student-t distribution.

    The Student-t with low degrees of freedom (df) produces heavy tails.
    As df -> inf, the distribution approaches Gaussian.

    Args:
        n_samples: Number of samples to generate.
        df: Degrees of freedom. Must be > 0. Lower values mean heavier tails.
        loc: Location (mean when df > 1).
        scale: Scale parameter (must be > 0).
        seed: Optional random seed for reproducibility.

    Returns:
        Array of shape (n_samples,) with Student-t samples.

    Raises:
        ValueError: If n_samples <= 0, df <= 0, or scale <= 0.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}.")
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}.")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}.")

    rng = np.random.default_rng(seed)
    samples = rng.standard_t(df=df, size=n_samples)
    return loc + scale * samples


def pareto(
    n_samples: int,
    alpha: float,
    x_min: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate samples from a Pareto (power-law) distribution.

    The Pareto distribution has a heavy right tail with tail index alpha.
    Smaller alpha => heavier tail.

    Args:
        n_samples: Number of samples to generate.
        alpha: Shape (tail index) parameter. Must be > 0.
        x_min: Scale (minimum value). Must be > 0.
        seed: Optional random seed for reproducibility.

    Returns:
        Array of shape (n_samples,) with Pareto samples.

    Raises:
        ValueError: If n_samples <= 0, alpha <= 0, or x_min <= 0.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}.")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}.")
    if x_min <= 0:
        raise ValueError(f"x_min must be positive, got {x_min}.")

    rng = np.random.default_rng(seed)
    # Use inverse CDF method: X = x_min / U^(1/alpha), U ~ Uniform(0,1)
    u = rng.uniform(0.0, 1.0, size=n_samples)
    return x_min / (u ** (1.0 / alpha))


def lognormal(
    n_samples: int,
    mu: float = 0.0,
    sigma: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate samples from a log-normal distribution.

    Log-normal distributions exhibit right-skewed, moderately heavy tails.

    Args:
        n_samples: Number of samples to generate.
        mu: Mean of the underlying normal (log-space).
        sigma: Standard deviation of the underlying normal (log-space). Must be > 0.
        seed: Optional random seed for reproducibility.

    Returns:
        Array of shape (n_samples,) with log-normal samples.

    Raises:
        ValueError: If n_samples <= 0 or sigma <= 0.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}.")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")

    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=mu, sigma=sigma, size=n_samples)


def skewed_student_t(
    n_samples: int,
    df: float,
    skewness: float = 1.0,
    loc: float = 0.0,
    scale: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate samples from a skewed Student-t distribution.

    Implements the Fernandez & Steel (1998) skewed-t parameterization.
    The skewness parameter gamma controls asymmetry:
        - gamma > 1: right-skewed (heavier right tail)
        - gamma < 1: left-skewed (heavier left tail)
        - gamma = 1: symmetric Student-t

    Sampling uses the two-piece scale method:
        - Sample u from Uniform(0,1)
        - If u < 1/(1+gamma^2), draw from the negative half; else from the positive half.

    Args:
        n_samples: Number of samples to generate.
        df: Degrees of freedom. Must be > 2 for finite variance.
        skewness: Asymmetry parameter gamma > 0. Default 1.0 gives symmetric t.
        loc: Location shift applied after sampling.
        scale: Scale parameter applied after sampling. Must be > 0.
        seed: Optional random seed for reproducibility.

    Returns:
        Array of shape (n_samples,) with skewed-t samples.

    Raises:
        ValueError: If n_samples <= 0, df <= 0, skewness <= 0, or scale <= 0.

    References:
        Fernandez, C., & Steel, M. F. J. (1998). On Bayesian modeling of fat tails
        and skewness. Journal of the American Statistical Association, 93(441), 359-371.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}.")
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}.")
    if skewness <= 0:
        raise ValueError(f"skewness (gamma) must be positive, got {skewness}.")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}.")

    rng = np.random.default_rng(seed)
    gamma = skewness

    # Draw standard t samples
    t_samples = rng.standard_t(df=df, size=n_samples)

    # Uniform indicator to assign to positive or negative pieces
    u = rng.uniform(0.0, 1.0, size=n_samples)
    threshold = 1.0 / (1.0 + gamma**2)

    # Two-piece scale: scale positive values by gamma, negative by 1/gamma
    skewed = np.where(u < threshold, t_samples / gamma, t_samples * gamma)

    return loc + scale * skewed


def generalized_pareto(
    n_samples: int,
    xi: float,
    beta: float = 1.0,
    mu: float = 0.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate samples from a Generalized Pareto Distribution (GPD).

    The GPD arises naturally as the limit distribution of threshold exceedances
    in Extreme Value Theory.

    Args:
        n_samples: Number of samples to generate.
        xi: Shape parameter (tail index). xi > 0 => heavy tail (Pareto-type),
            xi = 0 => exponential tail, xi < 0 => bounded tail.
        beta: Scale parameter. Must be > 0.
        mu: Location (threshold) parameter.
        seed: Optional random seed for reproducibility.

    Returns:
        Array of shape (n_samples,) with GPD samples.

    Raises:
        ValueError: If n_samples <= 0 or beta <= 0.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}.")
    if beta <= 0:
        raise ValueError(f"beta (scale) must be positive, got {beta}.")

    from scipy.stats import genpareto
    rng = np.random.default_rng(seed)
    return genpareto.rvs(c=xi, loc=mu, scale=beta, size=n_samples, random_state=rng)


def mixed_distribution(
    n_samples: int,
    gaussian_weight: float = 0.9,
    t_df: float = 3.0,
    mean: float = 0.0,
    std: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Generate samples from a mixture of Gaussian and Student-t distributions.

    Useful for simulating data where most observations follow a normal regime
    but extreme events follow a heavy-tailed process.

    Args:
        n_samples: Number of samples to generate.
        gaussian_weight: Fraction of samples drawn from Gaussian. Must be in [0, 1].
        t_df: Degrees of freedom for the Student-t component.
        mean: Mean for both components.
        std: Standard deviation scale for both components.
        seed: Optional random seed for reproducibility.

    Returns:
        Array of shape (n_samples,) with mixture samples.

    Raises:
        ValueError: If n_samples <= 0, gaussian_weight not in [0,1], or std <= 0.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}.")
    if not (0.0 <= gaussian_weight <= 1.0):
        raise ValueError(f"gaussian_weight must be in [0, 1], got {gaussian_weight}.")
    if std <= 0:
        raise ValueError(f"std must be positive, got {std}.")

    rng = np.random.default_rng(seed)
    n_gauss = int(n_samples * gaussian_weight)
    n_t = n_samples - n_gauss

    gauss_samples = rng.normal(loc=mean, scale=std, size=n_gauss)
    t_raw = rng.standard_t(df=t_df, size=n_t)
    t_samples = mean + std * t_raw

    combined = np.concatenate([gauss_samples, t_samples])
    rng.shuffle(combined)
    return combined
