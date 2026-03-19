"""
monte_carlo.py
--------------
Monte Carlo simulation from fitted parametric distributions.

Provides functions to simulate random paths and samples from fitted
distribution objects, enabling uncertainty quantification and scenario analysis.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def simulate_from_distribution(
    dist: Any,
    dist_params: dict[str, float],
    n_samples: int,
    n_simulations: int = 1,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Simulate random samples from a parametric distribution.

    Args:
        dist: Scipy frozen or unfrozen distribution with an ``rvs`` method.
        dist_params: Dictionary of parameters passed to ``dist.rvs``.
            For unfrozen dists (e.g. ``scipy.stats.norm``), pass ``loc``, ``scale``, etc.
            For frozen dists, pass ``{}``.
        n_samples: Number of observations per simulation.
        n_simulations: Number of independent simulation runs.
        seed: Optional random seed.

    Returns:
        Array of shape (n_simulations, n_samples) if n_simulations > 1,
        or (n_samples,) if n_simulations == 1.

    Raises:
        ValueError: If n_samples <= 0 or n_simulations <= 0.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}.")
    if n_simulations <= 0:
        raise ValueError(f"n_simulations must be positive, got {n_simulations}.")

    rng = np.random.default_rng(seed)
    # Generate all samples at once for efficiency
    samples = dist.rvs(size=(n_simulations, n_samples), random_state=rng, **dist_params)
    samples = np.asarray(samples, dtype=np.float64)

    return samples[0] if n_simulations == 1 else samples


def simulate_quantiles(
    dist: Any,
    dist_params: dict[str, float],
    probabilities: NDArray[np.float64],
    n_simulations: int = 1000,
    n_samples: int = 1000,
    seed: int | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Simulate quantile distributions via Monte Carlo.

    For each simulation, draws n_samples from the distribution and computes
    quantiles, yielding a sampling distribution of the quantile estimator.

    Args:
        dist: Scipy distribution.
        dist_params: Distribution parameters.
        probabilities: Array of probability levels (e.g., [0.95, 0.99, 0.999]).
        n_simulations: Number of Monte Carlo runs.
        n_samples: Sample size per run.
        seed: Optional random seed.

    Returns:
        Dictionary with:
            - ``probabilities``: The requested probability levels.
            - ``quantile_means``: Mean simulated quantile at each level.
            - ``quantile_stds``: Std dev of simulated quantiles.
            - ``quantile_ci_lower``: 2.5th percentile of simulated quantiles.
            - ``quantile_ci_upper``: 97.5th percentile of simulated quantiles.
            - ``theoretical_quantiles``: True theoretical quantiles.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    rng = np.random.default_rng(seed)

    # Matrix of shape (n_simulations, n_samples)
    all_samples = dist.rvs(size=(n_simulations, n_samples), random_state=rng, **dist_params)
    # Compute quantiles for each simulation run: shape (n_simulations, n_probs)
    sim_quantiles = np.quantile(all_samples, probs, axis=1).T  # (n_simulations, n_probs)

    theoretical_q = dist.ppf(probs, **dist_params)

    return {
        "probabilities": probs,
        "quantile_means": np.mean(sim_quantiles, axis=0),
        "quantile_stds": np.std(sim_quantiles, axis=0, ddof=1),
        "quantile_ci_lower": np.percentile(sim_quantiles, 2.5, axis=0),
        "quantile_ci_upper": np.percentile(sim_quantiles, 97.5, axis=0),
        "theoretical_quantiles": theoretical_q,
    }


def monte_carlo_tail_probability(
    threshold: float,
    dist: Any,
    dist_params: dict[str, float],
    n_simulations: int = 10_000,
    n_samples: int = 1000,
    seed: int | None = None,
) -> dict[str, float]:
    """Estimate P(X > threshold) via Monte Carlo simulation.

    Args:
        threshold: The extreme threshold value.
        dist: Scipy distribution.
        dist_params: Distribution parameters.
        n_simulations: Number of MC repetitions.
        n_samples: Sample size per repetition.
        seed: Optional random seed.

    Returns:
        Dictionary with:
            - ``mean_probability``: Mean estimated tail probability.
            - ``std_probability``: Standard deviation across simulations.
            - ``ci_lower``: 2.5th percentile.
            - ``ci_upper``: 97.5th percentile.
            - ``theoretical_probability``: Exact P(X > threshold) from dist.
    """
    rng = np.random.default_rng(seed)
    simulated = dist.rvs(size=(n_simulations, n_samples), random_state=rng, **dist_params)
    tail_probs = np.mean(simulated > threshold, axis=1)  # shape (n_simulations,)

    theoretical_p = float(1.0 - dist.cdf(threshold, **dist_params))

    return {
        "mean_probability": float(np.mean(tail_probs)),
        "std_probability": float(np.std(tail_probs, ddof=1)),
        "ci_lower": float(np.percentile(tail_probs, 2.5)),
        "ci_upper": float(np.percentile(tail_probs, 97.5)),
        "theoretical_probability": theoretical_p,
    }


def simulate_from_empirical(
    data: NDArray[np.float64],
    n_samples: int,
    n_simulations: int = 1,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Simulate by resampling from the empirical distribution (bootstrap).

    Args:
        data: Original data to resample from.
        n_samples: Number of samples per simulation.
        n_simulations: Number of bootstrap replications.
        seed: Optional random seed.

    Returns:
        Array of shape (n_simulations, n_samples) or (n_samples,) if n_simulations==1.
    """
    data = np.asarray(data, dtype=np.float64).flatten()
    rng = np.random.default_rng(seed)

    indices = rng.integers(0, len(data), size=(n_simulations, n_samples))
    samples = data[indices]

    return samples[0] if n_simulations == 1 else samples
