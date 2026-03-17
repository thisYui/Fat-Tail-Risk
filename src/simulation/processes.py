"""
simulation/processes.py
-----------------------
Các quá trình ngẫu nhiên (stochastic processes) để mô phỏng
chuỗi giá và return theo thời gian:

  - GBM  – Geometric Brownian Motion (Black-Scholes)
  - GBM với Jump Diffusion (Merton 1976)
  - GARCH(1,1) – mô phỏng volatility clustering
  - EGARCH(1,1) – asymmetric volatility
  - Heston Stochastic Volatility
  - Ornstein-Uhlenbeck (mean-reversion)
  - Fractional Brownian Motion (long memory, H ≠ 0.5)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Geometric Brownian Motion (GBM)
# ---------------------------------------------------------------------------

def gbm(
    n_steps: int,
    n_paths: int = 1,
    mu: float = 0.05,
    sigma: float = 0.20,
    S0: float = 100.0,
    dt: float = 1 / 252,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Mô phỏng Geometric Brownian Motion (lognormal diffusion).

    dS = μ·S·dt + σ·S·dW

    S(t) = S0 · exp[(μ - σ²/2)·t + σ·W(t)]

    Parameters
    ----------
    n_steps  : số bước thời gian
    n_paths  : số đường mô phỏng (paths)
    mu       : drift (annualised)
    sigma    : volatility (annualised)
    S0       : giá ban đầu
    dt       : độ dài bước thời gian (1/252 = 1 ngày giao dịch)

    Returns
    -------
    np.ndarray shape (n_steps+1, n_paths) – chuỗi giá
    """
    rng = _rng(seed)
    Z = rng.standard_normal(size=(n_steps, n_paths))
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_price = np.vstack([
        np.zeros((1, n_paths)),
        np.cumsum(log_returns, axis=0),
    ])
    return S0 * np.exp(log_price)


def gbm_returns(
    n_steps: int,
    n_paths: int = 1,
    mu: float = 0.05,
    sigma: float = 0.20,
    dt: float = 1 / 252,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sinh trực tiếp log-return của GBM (không cần tính giá).

    Returns
    -------
    np.ndarray shape (n_steps, n_paths)
    """
    rng = _rng(seed)
    Z = rng.standard_normal(size=(n_steps, n_paths))
    return (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z


# ---------------------------------------------------------------------------
# Jump Diffusion – Merton (1976)
# ---------------------------------------------------------------------------

def merton_jump_diffusion(
    n_steps: int,
    n_paths: int = 1,
    mu: float = 0.05,
    sigma: float = 0.20,
    lam: float = 10.0,
    jump_mu: float = -0.02,
    jump_sigma: float = 0.05,
    S0: float = 100.0,
    dt: float = 1 / 252,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Mô phỏng Merton Jump-Diffusion.

    dS/S = (μ - λ·κ)dt + σ·dW + J·dN

    Trong đó:
    - N ~ Poisson(λ·dt): số lần jump
    - J ~ LogNormal(jump_mu, jump_sigma²): biên độ jump
    - κ = E[J - 1] = exp(jump_mu + 0.5·jump_sigma²) - 1

    Parameters
    ----------
    lam        : cường độ jump (số jump trung bình / năm, vd 10)
    jump_mu    : log-mean của jump size
    jump_sigma : log-std của jump size
    """
    rng = _rng(seed)
    kappa = np.exp(jump_mu + 0.5 * jump_sigma**2) - 1
    drift_adj = (mu - lam * kappa - 0.5 * sigma**2) * dt

    # Diffusion component
    Z = rng.standard_normal(size=(n_steps, n_paths))
    diffusion = drift_adj + sigma * np.sqrt(dt) * Z

    # Jump component
    N_jumps = rng.poisson(lam=lam * dt, size=(n_steps, n_paths))
    J = rng.normal(loc=jump_mu, scale=jump_sigma, size=(n_steps, n_paths))
    jumps = N_jumps * J

    log_returns = diffusion + jumps
    log_price = np.vstack([
        np.zeros((1, n_paths)),
        np.cumsum(log_returns, axis=0),
    ])
    return S0 * np.exp(log_price)


# ---------------------------------------------------------------------------
# GARCH(1,1)
# ---------------------------------------------------------------------------

def garch_11(
    n_steps: int,
    n_paths: int = 1,
    mu: float = 0.0,
    omega: float = 1e-6,
    alpha: float = 0.09,
    beta: float = 0.90,
    sigma2_init: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mô phỏng GARCH(1,1).

    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    r_t  = μ + ε_t,   ε_t = σ_t · z_t,   z_t ~ N(0,1)

    Parameters
    ----------
    omega       : constant term ω (> 0)
    alpha       : ARCH term α (≥ 0)
    beta        : GARCH term β (≥ 0), α + β < 1 cho stationary
    sigma2_init : phương sai ban đầu (None → dùng unconditional variance)

    Returns
    -------
    returns   : np.ndarray shape (n_steps, n_paths)
    variances : np.ndarray shape (n_steps, n_paths) – conditional variance
    """
    rng = _rng(seed)
    if sigma2_init is None:
        sigma2_init = omega / max(1 - alpha - beta, 1e-8)

    returns = np.empty((n_steps, n_paths))
    variances = np.empty((n_steps, n_paths))

    sigma2 = np.full(n_paths, sigma2_init)
    Z = rng.standard_normal(size=(n_steps, n_paths))

    for t in range(n_steps):
        eps = np.sqrt(sigma2) * Z[t]
        returns[t] = mu + eps
        variances[t] = sigma2
        sigma2 = omega + alpha * eps**2 + beta * sigma2

    return returns, variances


def egarch_11(
    n_steps: int,
    n_paths: int = 1,
    mu: float = 0.0,
    omega: float = -0.1,
    alpha: float = 0.1,
    gamma: float = -0.1,
    beta: float = 0.97,
    log_sigma2_init: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mô phỏng EGARCH(1,1) – Nelson (1991).

    log(σ²_t) = ω + β·log(σ²_{t-1}) + α·(|z_{t-1}| - E[|z|]) + γ·z_{t-1}

    Tham số γ < 0 mô phỏng leverage effect (bad news → vol tăng mạnh hơn).

    Parameters
    ----------
    gamma : leverage parameter (thường âm)

    Returns
    -------
    returns      : np.ndarray (n_steps, n_paths)
    log_variances: np.ndarray (n_steps, n_paths)
    """
    rng = _rng(seed)
    E_abs_z = np.sqrt(2 / np.pi)  # E[|z|] với z ~ N(0,1)

    if log_sigma2_init is None:
        log_sigma2_init = omega / max(1 - beta, 1e-8)

    returns = np.empty((n_steps, n_paths))
    log_var = np.empty((n_steps, n_paths))

    log_sigma2 = np.full(n_paths, log_sigma2_init)
    Z = rng.standard_normal(size=(n_steps, n_paths))

    for t in range(n_steps):
        sigma = np.exp(0.5 * log_sigma2)
        eps = sigma * Z[t]
        returns[t] = mu + eps
        log_var[t] = log_sigma2
        log_sigma2 = (
            omega
            + beta * log_sigma2
            + alpha * (np.abs(Z[t]) - E_abs_z)
            + gamma * Z[t]
        )

    return returns, log_var


# ---------------------------------------------------------------------------
# Heston Stochastic Volatility
# ---------------------------------------------------------------------------

def heston(
    n_steps: int,
    n_paths: int = 1,
    mu: float = 0.05,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.3,
    rho: float = -0.7,
    V0: float = 0.04,
    S0: float = 100.0,
    dt: float = 1 / 252,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mô phỏng mô hình Heston (1993) bằng Euler-Maruyama.

    dS = μ·S·dt + √V·S·dW₁
    dV = κ(θ - V)dt + ξ·√V·dW₂
    Corr(dW₁, dW₂) = ρ

    Parameters
    ----------
    kappa : tốc độ hồi quy (mean-reversion speed)
    theta : long-run variance (long-run vol² = theta)
    xi    : vol of vol
    rho   : tương quan giữa return và vol shock (thường âm)
    V0    : phương sai ban đầu

    Returns
    -------
    prices    : np.ndarray (n_steps+1, n_paths)
    variances : np.ndarray (n_steps+1, n_paths)
    """
    rng = _rng(seed)
    prices = np.empty((n_steps + 1, n_paths))
    variances = np.empty((n_steps + 1, n_paths))
    prices[0] = S0
    variances[0] = V0

    Z1 = rng.standard_normal(size=(n_steps, n_paths))
    Z2 = rng.standard_normal(size=(n_steps, n_paths))
    W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2  # correlated Brownian

    V = np.full(n_paths, V0)
    S = np.full(n_paths, S0)

    for t in range(n_steps):
        sqrt_V = np.sqrt(np.maximum(V, 0))
        S = S * np.exp((mu - 0.5 * V) * dt + sqrt_V * np.sqrt(dt) * Z1[t])
        V = np.maximum(
            V + kappa * (theta - V) * dt + xi * sqrt_V * np.sqrt(dt) * W2[t],
            0.0,
        )
        prices[t + 1] = S
        variances[t + 1] = V

    return prices, variances


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck (mean-reverting)
# ---------------------------------------------------------------------------

def ornstein_uhlenbeck(
    n_steps: int,
    n_paths: int = 1,
    mu: float = 0.0,
    kappa: float = 1.0,
    sigma: float = 0.1,
    X0: float = 0.0,
    dt: float = 1 / 252,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Mô phỏng quá trình Ornstein-Uhlenbeck (OU).

    dX = κ(μ - X)dt + σ·dW

    Dùng để mô phỏng lãi suất, spread, hoặc log-volatility.

    Parameters
    ----------
    mu    : long-run mean (level of mean reversion)
    kappa : tốc độ hồi quy
    sigma : diffusion coefficient

    Returns
    -------
    np.ndarray shape (n_steps+1, n_paths)
    """
    rng = _rng(seed)
    X = np.empty((n_steps + 1, n_paths))
    X[0] = X0
    Z = rng.standard_normal(size=(n_steps, n_paths))

    for t in range(n_steps):
        X[t + 1] = (
            X[t]
            + kappa * (mu - X[t]) * dt
            + sigma * np.sqrt(dt) * Z[t]
        )
    return X


# ---------------------------------------------------------------------------
# Fractional Brownian Motion (fBM)
# ---------------------------------------------------------------------------

def fractional_brownian_motion(
    n_steps: int,
    n_paths: int = 1,
    H: float = 0.7,
    sigma: float = 1.0,
    dt: float = 1 / 252,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Mô phỏng Fractional Brownian Motion bằng phương pháp Cholesky.

    H = 0.5  → Standard BM (no memory)
    H > 0.5  → Long memory (persistence, trending)
    H < 0.5  → Anti-persistence (mean-reverting)

    Parameters
    ----------
    H     : Hurst exponent ∈ (0, 1)
    sigma : scale

    Returns
    -------
    np.ndarray shape (n_steps+1, n_paths) – fBM paths

    Notes
    -----
    Cholesky decomposition có độ phức tạp O(n²) – dùng cho n nhỏ/vừa.
    """
    rng = _rng(seed)
    n = n_steps + 1
    t = np.arange(n) * dt

    # Covariance matrix của fBM
    T1, T2 = np.meshgrid(t, t)
    cov = 0.5 * sigma**2 * (
        np.abs(T1) ** (2 * H)
        + np.abs(T2) ** (2 * H)
        - np.abs(T1 - T2) ** (2 * H)
    )

    # Add small jitter for numerical stability
    cov += np.eye(n) * 1e-10

    L = np.linalg.cholesky(cov)
    Z = rng.standard_normal(size=(n, n_paths))
    return L @ Z