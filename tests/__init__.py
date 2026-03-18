"""
tests/__init__.py
------------------
Shared fixtures và helpers dùng chung cho toàn bộ test suite.

Cung cấp
--------
  - make_normal_returns()   : synthetic Normal returns
  - make_fat_tail_returns() : synthetic Student-t fat-tail returns
  - make_garch_returns()    : synthetic GARCH(1,1) returns
  - make_price_series()     : synthetic price series
  - SEEDS, TOL              : constants
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED  = 42
TOL   = 1e-8       # numerical tolerance cho equality checks
RTOL  = 1e-3       # relative tolerance cho approximate checks
N_OBS = 2_000      # default số quan sát synthetic


# ---------------------------------------------------------------------------
# Shared synthetic data generators
# ---------------------------------------------------------------------------

def make_normal_returns(
    n: int = N_OBS,
    mu: float = 0.0005,
    sigma: float = 0.01,
    seed: int = SEED,
) -> np.ndarray:
    """Sinh chuỗi return từ phân phối Normal."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mu, scale=sigma, size=n)


def make_fat_tail_returns(
    n: int = N_OBS,
    df: float = 4.0,
    mu: float = 0.0,
    sigma: float = 0.01,
    seed: int = SEED,
) -> np.ndarray:
    """Sinh chuỗi return từ Student-t (fat-tail)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_t(df=df, size=n)
    # Standardise scale
    from scipy import stats
    scale = np.sqrt(df / (df - 2)) if df > 2 else 1.0
    return mu + sigma * z / scale


def make_garch_returns(
    n: int = N_OBS,
    mu: float = 0.0,
    omega: float = 1e-6,
    alpha: float = 0.09,
    beta: float = 0.90,
    seed: int = SEED,
) -> np.ndarray:
    """Sinh chuỗi return từ GARCH(1,1)."""
    rng = np.random.default_rng(seed)
    returns = np.empty(n)
    sigma2  = omega / max(1 - alpha - beta, 1e-8)
    z = rng.standard_normal(n)
    for t in range(n):
        eps = np.sqrt(sigma2) * z[t]
        returns[t] = mu + eps
        sigma2 = omega + alpha * eps**2 + beta * sigma2
    return returns


def make_price_series(
    n: int = N_OBS,
    mu: float = 0.0005,
    sigma: float = 0.01,
    S0: float = 100.0,
    seed: int = SEED,
) -> pd.Series:
    """Sinh chuỗi giá từ GBM."""
    r = make_normal_returns(n=n, mu=mu, sigma=sigma, seed=seed)
    prices = S0 * np.exp(np.cumsum(r))
    dates  = pd.date_range("2010-01-01", periods=n, freq="B")
    return pd.Series(prices, index=dates, name="price")


def make_return_series(
    n: int = N_OBS,
    mu: float = 0.0005,
    sigma: float = 0.01,
    seed: int = SEED,
) -> pd.Series:
    """Sinh pd.Series return có DatetimeIndex."""
    r     = make_normal_returns(n=n, mu=mu, sigma=sigma, seed=seed)
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    return pd.Series(r, index=dates, name="return")


def make_multi_returns(
    n: int = N_OBS,
    d: int = 3,
    corr: float = 0.4,
    seed: int = SEED,
) -> np.ndarray:
    """Sinh ma trận return đa tài sản (n × d) với tương quan."""
    rng  = np.random.default_rng(seed)
    cov  = np.full((d, d), corr)
    np.fill_diagonal(cov, 1.0)
    L    = np.linalg.cholesky(cov)
    z    = rng.standard_normal((n, d)) @ L.T
    return z * 0.01  # scale về daily vol ~1%