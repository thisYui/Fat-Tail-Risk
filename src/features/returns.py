"""
features/returns.py
-------------------
Tính toán các loại return từ chuỗi giá hoặc NAV:
  - Simple return
  - Log return
  - Excess return (so với benchmark / risk-free rate)
  - Rolling return
  - Annualised return
  - Cumulative return
  - Drawdown
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Union

ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame]


# ---------------------------------------------------------------------------
# Core return calculations
# ---------------------------------------------------------------------------

def simple_returns(prices: ArrayLike, periods: int = 1) -> ArrayLike:
    """
    Tính simple (arithmetic) return: (P_t - P_{t-k}) / P_{t-k}.

    Parameters
    ----------
    prices  : chuỗi giá / NAV (Series hoặc DataFrame)
    periods : độ trễ k (default 1 – daily return)
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    return prices.pct_change(periods=periods)


def log_returns(prices: ArrayLike, periods: int = 1) -> ArrayLike:
    """
    Tính log (continuous compound) return: ln(P_t / P_{t-k}).

    Parameters
    ----------
    prices  : chuỗi giá / NAV
    periods : độ trễ k (default 1)
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    return np.log(prices / prices.shift(periods))


def excess_returns(
    returns: ArrayLike,
    benchmark: Union[float, ArrayLike] = 0.0,
) -> ArrayLike:
    """
    Tính excess return so với benchmark hoặc risk-free rate.

    Parameters
    ----------
    returns   : chuỗi return của tài sản
    benchmark : float (annualised risk-free) hoặc Series return của benchmark.
                Nếu là float, tự động chuyển sang daily bằng cách chia 252.
    """
    if isinstance(benchmark, float):
        daily_rf = benchmark / 252
        return returns - daily_rf
    return returns - benchmark


def rolling_returns(
    prices: ArrayLike,
    window: int,
    log: bool = False,
) -> ArrayLike:
    """
    Tính rolling return qua cửa sổ `window` phiên.

    Parameters
    ----------
    prices  : chuỗi giá
    window  : số phiên trong cửa sổ
    log     : True → log return, False → simple return
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    if log:
        return np.log(prices / prices.shift(window))
    return prices.pct_change(periods=window)


def annualised_return(
    returns: ArrayLike,
    freq: int = 252,
    method: str = "geometric",
) -> float:
    """
    Tính annualised return từ chuỗi return định kỳ.

    Parameters
    ----------
    returns : chuỗi periodic return (đã drop NaN)
    freq    : số kỳ trong 1 năm (252=daily, 52=weekly, 12=monthly)
    method  : 'geometric' (default) hoặc 'arithmetic'
    """
    r = pd.Series(returns).dropna()
    if method == "geometric":
        cumulative = (1 + r).prod()
        n_years = len(r) / freq
        return float(cumulative ** (1 / n_years) - 1)
    return float(r.mean() * freq)


# ---------------------------------------------------------------------------
# Cumulative & Drawdown
# ---------------------------------------------------------------------------

def cumulative_returns(returns: ArrayLike, start_value: float = 1.0) -> ArrayLike:
    """
    Tính cumulative return (growth of $1).

    Parameters
    ----------
    returns     : chuỗi periodic return
    start_value : giá trị ban đầu (default 1.0)
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)
    return (1 + returns.fillna(0)).cumprod() * start_value


def drawdown(prices_or_cum: ArrayLike) -> pd.Series:
    """
    Tính drawdown so với high-water mark: (P_t - HWM_t) / HWM_t.

    Returns
    -------
    pd.Series – giá trị âm hoặc 0
    """
    s = pd.Series(prices_or_cum) if isinstance(prices_or_cum, np.ndarray) else prices_or_cum
    hwm = s.cummax()
    return (s - hwm) / hwm


def max_drawdown(prices_or_cum: ArrayLike) -> float:
    """Trả về Maximum Drawdown (giá trị âm nhất)."""
    return float(drawdown(prices_or_cum).min())


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def return_summary(
    prices: ArrayLike,
    freq: int = 252,
    risk_free: float = 0.0,
) -> pd.Series:
    """
    Tổng hợp các chỉ số return chính vào một Series.

    Parameters
    ----------
    prices     : chuỗi giá
    freq       : số kỳ / năm
    risk_free  : lãi suất phi rủi ro annualised (default 0)

    Returns
    -------
    pd.Series: total_return, annualised_return, annualised_vol,
               sharpe_ratio, max_drawdown
    """
    r = log_returns(prices).dropna()
    ann_ret = annualised_return(r, freq=freq)
    ann_vol = float(r.std() * np.sqrt(freq))
    excess = annualised_return(excess_returns(r, risk_free), freq=freq)
    sharpe = excess / ann_vol if ann_vol != 0 else np.nan
    cum = cumulative_returns(r)
    total_ret = float(cum.iloc[-1] - 1)

    return pd.Series(
        {
            "total_return": total_ret,
            "annualised_return": ann_ret,
            "annualised_vol": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown(cum),
        }
    )