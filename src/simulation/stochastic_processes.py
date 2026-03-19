"""
stochastic_processes.py
-----------------------
Stochastic process simulation for fat-tail modeling.

Implements:
    - Geometric Brownian Motion (GBM): standard continuous-time model.
    - GBM with Student-t innovations: heavy-tailed discrete-time alternative.
    - Ornstein-Uhlenbeck (OU) process: mean-reverting process.
    - Jump-diffusion process: GBM with Poisson-distributed jumps.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def geometric_brownian_motion(
    n_steps: int,
    n_paths: int = 1,
    mu: float = 0.0,
    sigma: float = 1.0,
    dt: float = 1.0 / 252,
    x0: float = 100.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Simulate Geometric Brownian Motion (GBM) paths.

    The GBM follows:
        dX_t = mu * X_t * dt + sigma * X_t * dW_t

    Exact discrete solution:
        X_{t+dt} = X_t * exp( (mu - sigma^2/2) * dt + sigma * sqrt(dt) * Z_t )
    where Z_t ~ N(0,1).

    Args:
        n_steps: Number of time steps.
        n_paths: Number of independent paths to simulate.
        mu: Drift rate (annualized).
        sigma: Volatility (annualized). Must be > 0.
        dt: Time step size (fraction of year). Default = 1/252 (daily).
        x0: Initial value of the process.
        seed: Optional random seed.

    Returns:
        Array of shape (n_paths, n_steps + 1) with simulated paths including
        the initial value at index 0.

    Raises:
        ValueError: If sigma <= 0, n_steps <= 0, or n_paths <= 0.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}.")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}.")

    rng = np.random.default_rng(seed)
    # Standard Gaussian innovations
    z = rng.standard_normal(size=(n_paths, n_steps))
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z

    # Cumulative sum to build log-price paths; prepend 0 for initial value
    log_price = np.concatenate([
        np.zeros((n_paths, 1)),
        np.cumsum(log_returns, axis=1)
    ], axis=1)

    paths = x0 * np.exp(log_price)
    return paths[0] if n_paths == 1 else paths


def gbm_with_t_innovations(
    n_steps: int,
    n_paths: int = 1,
    mu: float = 0.0,
    sigma: float = 1.0,
    df: float = 5.0,
    dt: float = 1.0 / 252,
    x0: float = 100.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Simulate GBM with heavy-tailed (Student-t) innovations.

    Replaces the Gaussian increments of standard GBM with scaled t-distributed
    increments. As df -> inf, this converges to standard GBM. With df ~ 3-5,
    it captures the excess kurtosis observed in many real datasets.

    Innovation at each step: sqrt(dt) * sigma * Z_t, where Z_t ~ t(df) / sqrt(df/(df-2))
    (normalized to have unit variance for finite df > 2).

    Args:
        n_steps: Number of time steps.
        n_paths: Number of independent paths.
        mu: Drift parameter.
        sigma: Scale of innovations. Must be > 0.
        df: Degrees of freedom for t innovations. Must be > 2 for finite variance.
        dt: Time step size.
        x0: Initial value.
        seed: Optional random seed.

    Returns:
        Array of shape (n_paths, n_steps + 1) or (n_steps + 1,) if n_paths == 1.

    Raises:
        ValueError: If df <= 2 (infinite variance), sigma <= 0, or n_steps/n_paths <= 0.
    """
    if df <= 2:
        raise ValueError(f"df must be > 2 for finite variance t innovations, got {df}.")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}.")

    rng = np.random.default_rng(seed)
    # t innovation with unit variance: t(df) / sqrt(df/(df-2))
    t_raw = rng.standard_t(df=df, size=(n_paths, n_steps))
    t_normalized = t_raw / np.sqrt(df / (df - 2))

    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * t_normalized

    log_price = np.concatenate([
        np.zeros((n_paths, 1)),
        np.cumsum(log_returns, axis=1)
    ], axis=1)

    paths = x0 * np.exp(log_price)
    return paths[0] if n_paths == 1 else paths


def ornstein_uhlenbeck(
    n_steps: int,
    n_paths: int = 1,
    theta: float = 0.1,
    mu: float = 0.0,
    sigma: float = 1.0,
    dt: float = 1.0,
    x0: float = 0.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Simulate an Ornstein-Uhlenbeck (OU) mean-reverting process.

    The OU process follows:
        dX_t = theta * (mu - X_t) * dt + sigma * dW_t

    Exact discrete scheme:
        X_{t+dt} = X_t * exp(-theta * dt) + mu * (1 - exp(-theta * dt))
                   + sigma * sqrt((1 - exp(-2*theta*dt)) / (2*theta)) * Z_t

    Args:
        n_steps: Number of time steps.
        n_paths: Number of independent paths.
        theta: Mean reversion speed. Must be > 0.
        mu: Long-run mean.
        sigma: Diffusion coefficient. Must be > 0.
        dt: Time step size.
        x0: Initial value.
        seed: Optional random seed.

    Returns:
        Array of shape (n_paths, n_steps + 1) or (n_steps + 1,) if n_paths == 1.

    Raises:
        ValueError: If theta <= 0 or sigma <= 0.
    """
    if theta <= 0:
        raise ValueError(f"theta must be positive, got {theta}.")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")

    rng = np.random.default_rng(seed)

    e = np.exp(-theta * dt)
    # Conditional variance of one step
    cond_var = sigma**2 * (1 - e**2) / (2 * theta)
    cond_std = np.sqrt(cond_var)

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = x0
    z = rng.standard_normal(size=(n_paths, n_steps))

    for t in range(n_steps):
        paths[:, t + 1] = mu + (paths[:, t] - mu) * e + cond_std * z[:, t]

    return paths[0] if n_paths == 1 else paths


def jump_diffusion(
    n_steps: int,
    n_paths: int = 1,
    mu: float = 0.0,
    sigma: float = 0.2,
    jump_intensity: float = 5.0,
    jump_mean: float = 0.0,
    jump_std: float = 0.1,
    dt: float = 1.0 / 252,
    x0: float = 100.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Simulate a Merton (1976) jump-diffusion process.

    Combines GBM with a compound Poisson jump component:
        dX_t = (mu - lambda*kappa) * X_t * dt + sigma * X_t * dW_t + X_t * (e^J - 1) * dN_t

    where N_t is a Poisson process with intensity lambda, and J ~ N(jump_mean, jump_std^2).

    Args:
        n_steps: Number of time steps.
        n_paths: Number of independent paths.
        mu: Drift rate.
        sigma: Diffusion volatility. Must be > 0.
        jump_intensity: Poisson jump intensity (expected jumps per unit time). Must be >= 0.
        jump_mean: Mean of log-jump size (J ~ Normal).
        jump_std: Std dev of log-jump size. Must be >= 0.
        dt: Time step size.
        x0: Initial process value.
        seed: Optional random seed.

    Returns:
        Array of shape (n_paths, n_steps + 1) or (n_steps + 1,) if n_paths == 1.

    Raises:
        ValueError: If sigma <= 0 or jump_intensity < 0.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    if jump_intensity < 0:
        raise ValueError(f"jump_intensity must be >= 0, got {jump_intensity}.")

    rng = np.random.default_rng(seed)

    # Correction for risk-neutral drift: kappa = E[e^J - 1]
    kappa = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0

    # GBM component
    z_diff = rng.standard_normal(size=(n_paths, n_steps))
    # Poisson jump counts
    n_jumps = rng.poisson(lam=jump_intensity * dt, size=(n_paths, n_steps))
    # Jump sizes: sum of n_jump draws from N(jump_mean, jump_std^2)
    max_jumps = int(n_jumps.max()) + 1
    jump_sizes_all = rng.normal(loc=jump_mean, scale=max(jump_std, 1e-10),
                                size=(n_paths, n_steps, max_jumps))
    # Only count as many jumps as occurred
    jump_component = np.where(
        n_jumps[:, :, np.newaxis] > np.arange(max_jumps)[np.newaxis, np.newaxis, :],
        jump_sizes_all, 0.0,
    ).sum(axis=-1)  # shape (n_paths, n_steps)

    log_returns = (
        (mu - jump_intensity * kappa - 0.5 * sigma**2) * dt
        + sigma * np.sqrt(dt) * z_diff
        + jump_component
    )

    log_price = np.concatenate([
        np.zeros((n_paths, 1)),
        np.cumsum(log_returns, axis=1)
    ], axis=1)

    paths = x0 * np.exp(log_price)
    return paths[0] if n_paths == 1 else paths
