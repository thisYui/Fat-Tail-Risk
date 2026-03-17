"""
features/moments.py
--------------------
Tính các thống kê moment của phân phối return:
  - Mean, Variance, Std
  - Skewness (độ lệch)
  - Kurtosis / Excess Kurtosis
  - Rolling moments
  - Jarque-Bera test (kiểm định chuẩn)
  - Moment summary
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union

ArrayLike = Union[np.ndarray, pd.Series]


# ---------------------------------------------------------------------------
# Basic moments
# ---------------------------------------------------------------------------

def mean(returns: ArrayLike, annualise: bool = False, freq: int = 252) -> float:
    """
    Tính trung bình return.

    Parameters
    ----------
    returns  : chuỗi return
    annualise: True → nhân với freq
    freq     : số kỳ / năm (252 = daily)
    """
    r = pd.Series(returns).dropna()
    m = float(r.mean())
    return m * freq if annualise else m


def variance(returns: ArrayLike, annualise: bool = False, freq: int = 252) -> float:
    """
    Tính variance của return.

    Parameters
    ----------
    annualise : True → nhân với freq (giả sử i.i.d.)
    """
    r = pd.Series(returns).dropna()
    v = float(r.var())
    return v * freq if annualise else v


def std(returns: ArrayLike, annualise: bool = False, freq: int = 252) -> float:
    """
    Tính độ lệch chuẩn (volatility).

    Parameters
    ----------
    annualise : True → nhân với sqrt(freq)
    """
    r = pd.Series(returns).dropna()
    s = float(r.std())
    return s * np.sqrt(freq) if annualise else s


def skewness(returns: ArrayLike) -> float:
    """
    Tính skewness (độ lệch).
    - Dương → đuôi phải dài (right-skewed)
    - Âm  → đuôi trái dài (left-skewed, phổ biến với tài sản tài chính)
    """
    r = pd.Series(returns).dropna()
    return float(stats.skew(r))


def kurtosis(returns: ArrayLike, excess: bool = True) -> float:
    """
    Tính kurtosis của phân phối.

    Parameters
    ----------
    excess : True (default) → excess kurtosis = kurtosis - 3
             (phân phối chuẩn = 0; fat-tail > 0)

    Notes
    -----
    Scipy dùng excess kurtosis theo mặc định (fisher=True).
    """
    r = pd.Series(returns).dropna()
    k = float(stats.kurtosis(r, fisher=excess))
    return k


def semi_deviation(returns: ArrayLike, threshold: float = 0.0) -> float:
    """
    Tính semi-deviation (độ lệch chuẩn của các return âm dưới ngưỡng).

    Parameters
    ----------
    threshold : ngưỡng (default 0 → chỉ xét return âm)

    Returns
    -------
    float – semi-deviation (downside risk)
    """
    r = pd.Series(returns).dropna()
    downside = r[r < threshold]
    if len(downside) == 0:
        return 0.0
    return float(downside.std())


# ---------------------------------------------------------------------------
# Rolling moments
# ---------------------------------------------------------------------------

def rolling_mean(returns: ArrayLike, window: int) -> pd.Series:
    """Rolling mean return."""
    return pd.Series(returns).rolling(window).mean()


def rolling_std(returns: ArrayLike, window: int) -> pd.Series:
    """Rolling volatility (std)."""
    return pd.Series(returns).rolling(window).std()


def rolling_skewness(returns: ArrayLike, window: int) -> pd.Series:
    """Rolling skewness."""
    return pd.Series(returns).rolling(window).skew()


def rolling_kurtosis(returns: ArrayLike, window: int) -> pd.Series:
    """Rolling excess kurtosis."""
    return pd.Series(returns).rolling(window).kurt()


# ---------------------------------------------------------------------------
# Normality test
# ---------------------------------------------------------------------------

def jarque_bera_test(returns: ArrayLike) -> dict:
    """
    Jarque-Bera test kiểm định phân phối chuẩn.

    Returns
    -------
    dict với keys:
        statistic  : JB statistic
        p_value    : p-value
        is_normal  : True nếu p_value > 0.05 (không bác bỏ H0 chuẩn)
        skewness   : skewness mẫu
        kurtosis   : excess kurtosis mẫu
    """
    r = pd.Series(returns).dropna()
    jb_stat, p_val = stats.jarque_bera(r)
    return {
        "statistic": float(jb_stat),
        "p_value": float(p_val),
        "is_normal": p_val > 0.05,
        "skewness": skewness(r),
        "kurtosis": kurtosis(r, excess=True),
    }


def shapiro_wilk_test(returns: ArrayLike) -> dict:
    """
    Shapiro-Wilk test kiểm định chuẩn (phù hợp n < 5000).

    Returns
    -------
    dict với keys: statistic, p_value, is_normal
    """
    r = pd.Series(returns).dropna()
    stat, p_val = stats.shapiro(r)
    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "is_normal": p_val > 0.05,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def moments_summary(
    returns: ArrayLike,
    freq: int = 252,
) -> pd.Series:
    """
    Tổng hợp tất cả các moment statistics vào một Series.

    Parameters
    ----------
    returns : chuỗi return
    freq    : số kỳ / năm để annualise

    Returns
    -------
    pd.Series bao gồm:
        mean, annualised_mean, std, annualised_vol,
        skewness, excess_kurtosis, semi_deviation,
        jb_statistic, jb_pvalue, is_normal
    """
    r = pd.Series(returns).dropna()
    jb = jarque_bera_test(r)

    return pd.Series(
        {
            "mean": mean(r),
            "annualised_mean": mean(r, annualise=True, freq=freq),
            "std": std(r),
            "annualised_vol": std(r, annualise=True, freq=freq),
            "skewness": skewness(r),
            "excess_kurtosis": kurtosis(r, excess=True),
            "semi_deviation": semi_deviation(r),
            "jb_statistic": jb["statistic"],
            "jb_pvalue": jb["p_value"],
            "is_normal": jb["is_normal"],
        }
    )