"""
risk/backtest.py
-----------------
Backtesting framework cho mô hình VaR / CVaR:

  - VaR backtesting: Kupiec POF test, Christoffersen interval test
  - CVaR / ES backtesting: McNeil-Frey test, Bootstrap test
  - Traffic light system (Basel)
  - Rolling backtest engine
  - BacktestResult & BacktestReport dataclasses
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """
    Kết quả backtest của một mô hình VaR.

    Attributes
    ----------
    model_name        : tên mô hình
    confidence        : mức tin cậy
    n_obs             : số quan sát
    n_violations      : số lần return < -VaR (violations)
    violation_rate    : tỷ lệ violation thực tế
    expected_rate     : tỷ lệ violation kỳ vọng (= 1 - confidence)
    kupiec_stat       : Kupiec POF test statistic
    kupiec_pvalue     : p-value Kupiec test
    christoffersen_stat  : Christoffersen independence test statistic
    christoffersen_pvalue: p-value
    traffic_light     : 'green' / 'yellow' / 'red' (Basel)
    pass_kupiec       : True nếu model không bị bác bỏ (p > 0.05)
    """
    model_name: str
    confidence: float
    n_obs: int
    n_violations: int
    violation_rate: float
    expected_rate: float
    kupiec_stat: float
    kupiec_pvalue: float
    christoffersen_stat: float
    christoffersen_pvalue: float
    traffic_light: str
    pass_kupiec: bool
    pass_christoffersen: bool
    violations_index: Optional[np.ndarray] = None

    def as_series(self) -> pd.Series:
        return pd.Series({
            "model": self.model_name,
            "confidence": self.confidence,
            "n_obs": self.n_obs,
            "n_violations": self.n_violations,
            "violation_rate": self.violation_rate,
            "expected_rate": self.expected_rate,
            "kupiec_stat": self.kupiec_stat,
            "kupiec_pvalue": self.kupiec_pvalue,
            "chr_stat": self.christoffersen_stat,
            "chr_pvalue": self.christoffersen_pvalue,
            "traffic_light": self.traffic_light,
            "pass_kupiec": self.pass_kupiec,
            "pass_chr": self.pass_christoffersen,
        })


@dataclass
class BacktestReport:
    """Báo cáo so sánh nhiều mô hình VaR."""
    results: List[BacktestResult]
    confidence: float
    n_obs: int

    def summary(self) -> pd.DataFrame:
        rows = [r.as_series() for r in self.results]
        return pd.DataFrame(rows).set_index("model")

    def best_model(self) -> str:
        """Model vượt cả hai test với p-value cao nhất (Kupiec)."""
        passing = [r for r in self.results if r.pass_kupiec and r.pass_christoffersen]
        if not passing:
            passing = self.results
        best = max(passing, key=lambda r: r.kupiec_pvalue)
        return best.model_name


# ---------------------------------------------------------------------------
# Violation sequence
# ---------------------------------------------------------------------------

def compute_violations(
    returns: np.ndarray,
    var_forecasts: np.ndarray,
) -> np.ndarray:
    """
    Tính chuỗi violation indicator I_t = 1{R_t < -VaR_t}.

    Parameters
    ----------
    returns       : chuỗi return thực tế (n,)
    var_forecasts : chuỗi VaR forecast (n,), giá trị âm

    Returns
    -------
    np.ndarray bool (n,) – True = violation
    """
    r = np.asarray(returns)
    var = np.asarray(var_forecasts)
    return r < var   # VaR đã là giá trị âm


# ---------------------------------------------------------------------------
# Kupiec POF (Proportion of Failures) Test
# ---------------------------------------------------------------------------

def kupiec_test(
    violations: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Kupiec (1995) Proportion of Failures test.

    H0: violation rate = (1 - confidence) [mô hình calibrated đúng]
    H1: violation rate ≠ (1 - confidence)

    Likelihood Ratio statistic ~ χ²(1) dưới H0.

    Parameters
    ----------
    violations : bool array – I_t indicator
    confidence : mức tin cậy của VaR

    Returns
    -------
    Tuple[float, float] – (LR_stat, p_value)
    """
    n = len(violations)
    x = int(np.sum(violations))    # số violations
    p = 1 - confidence             # expected rate
    p_hat = x / n                  # observed rate

    if x == 0 or x == n:
        # Tránh log(0)
        return np.inf, 0.0

    # LR = -2 ln[p^x (1-p)^(n-x)] + 2 ln[p_hat^x (1-p_hat)^(n-x)]
    lr = -2 * (
        x * np.log(p) + (n - x) * np.log(1 - p)
        - x * np.log(p_hat) - (n - x) * np.log(1 - p_hat)
    )
    p_val = float(1 - stats.chi2.cdf(lr, df=1))
    return float(lr), p_val


# ---------------------------------------------------------------------------
# Christoffersen Independence Test
# ---------------------------------------------------------------------------

def christoffersen_test(
    violations: np.ndarray,
) -> Tuple[float, float]:
    """
    Christoffersen (1998) Independence test.

    Kiểm định xem violations có clustered hay không.

    H0: violations are i.i.d. (no autocorrelation)

    Returns
    -------
    Tuple[float, float] – (LR_ind, p_value)
    """
    v = np.asarray(violations, dtype=int)
    n = len(v)

    # Transition counts
    n00 = np.sum((v[:-1] == 0) & (v[1:] == 0))
    n01 = np.sum((v[:-1] == 0) & (v[1:] == 1))
    n10 = np.sum((v[:-1] == 1) & (v[1:] == 0))
    n11 = np.sum((v[:-1] == 1) & (v[1:] == 1))

    # Transition probabilities
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n, 1)

    if pi in (0, 1) or pi01 in (0, 1) or pi11 in (0, 1):
        return np.nan, np.nan

    # LR statistic
    ll_unres = (
        n01 * np.log(pi01) + n00 * np.log(1 - pi01)
        + n11 * np.log(pi11) + n10 * np.log(1 - pi11)
    )
    ll_res = (n01 + n11) * np.log(pi) + (n00 + n10) * np.log(1 - pi)
    lr = -2 * (ll_res - ll_unres)
    p_val = float(1 - stats.chi2.cdf(lr, df=1))
    return float(lr), p_val


# ---------------------------------------------------------------------------
# Christoffersen Conditional Coverage (POF + Independence)
# ---------------------------------------------------------------------------

def conditional_coverage_test(
    violations: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Christoffersen Conditional Coverage Test = POF + Independence.

    LR_cc = LR_uc + LR_ind ~ χ²(2)

    Returns
    -------
    Tuple[float, float] – (LR_cc, p_value)
    """
    lr_uc, _ = kupiec_test(violations, confidence)
    lr_ind, _ = christoffersen_test(violations)
    if np.isnan(lr_ind):
        return np.nan, np.nan
    lr_cc = lr_uc + lr_ind
    p_val = float(1 - stats.chi2.cdf(lr_cc, df=2))
    return float(lr_cc), p_val


# ---------------------------------------------------------------------------
# Basel Traffic Light System
# ---------------------------------------------------------------------------

def basel_traffic_light(
    n_violations: int,
    n_obs: int = 250,
    confidence: float = 0.99,
) -> str:
    """
    Hệ thống đèn giao thông Basel II/III cho VaR 99% trên 250 ngày.

    Green  : 0-4 violations → multiplier = 3
    Yellow : 5-9 violations → multiplier = 3.4-3.85
    Red    : 10+ violations → multiplier = 4

    Parameters
    ----------
    n_violations : số violations
    n_obs        : số ngày backtest (thường 250)
    confidence   : mức tin cậy (thường 0.99)

    Returns
    -------
    str – 'green', 'yellow', hoặc 'red'
    """
    expected = int((1 - confidence) * n_obs)
    # Basel thresholds (designed for n=250, p=0.01)
    green_max = max(4, expected * 2)
    yellow_max = max(9, expected * 4)

    if n_violations <= green_max:
        return "green"
    elif n_violations <= yellow_max:
        return "yellow"
    else:
        return "red"


# ---------------------------------------------------------------------------
# ES / CVaR Backtest (McNeil-Frey)
# ---------------------------------------------------------------------------

def mcneil_frey_test(
    returns: np.ndarray,
    var_forecasts: np.ndarray,
    es_forecasts: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    McNeil-Frey (2000) test cho Expected Shortfall.

    Kiểm định H0: E[R_t | R_t < -VaR_t] = ES_t

    Dùng standardised residuals: Z_t = (R_t - ES_t) / σ_t
    Test xem E[Z_t | violation] = 0 bằng t-test.

    Parameters
    ----------
    returns       : return thực tế (n,)
    var_forecasts : VaR forecast (n,), giá trị âm
    es_forecasts  : ES forecast (n,), giá trị âm

    Returns
    -------
    Tuple[float, float] – (t_stat, p_value)
    """
    r = np.asarray(returns)
    var = np.asarray(var_forecasts)
    es = np.asarray(es_forecasts)

    violations = r < var
    if violations.sum() < 5:
        return np.nan, np.nan

    # Standardised exceedances
    exceedances = r[violations]
    es_at_violation = es[violations]
    sigma = np.std(exceedances)
    if sigma == 0:
        return np.nan, np.nan

    z = (exceedances - es_at_violation) / sigma
    t_stat, p_val = stats.ttest_1samp(z, popmean=0)
    return float(t_stat), float(p_val)


def es_bootstrap_test(
    returns: np.ndarray,
    es_forecasts: np.ndarray,
    var_forecasts: np.ndarray,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Bootstrap test cho ES accuracy.

    Tính test statistic T = mean(R_t | violation) - mean(ES_t | violation)
    Bootstrap distribution dưới H0.

    Returns
    -------
    Tuple[float, float] – (test_stat, p_value)
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(returns)
    var = np.asarray(var_forecasts)
    es = np.asarray(es_forecasts)

    violations = r < var
    if violations.sum() < 5:
        return np.nan, np.nan

    # Observed statistic
    obs_stat = float(
        np.mean(r[violations]) - np.mean(es[violations])
    )

    # Bootstrap under H0 (resample residuals)
    residuals = r[violations] - es[violations]
    boot_stats = []
    for _ in range(n_bootstrap):
        boot_res = rng.choice(residuals, size=len(residuals), replace=True)
        boot_stats.append(float(np.mean(boot_res)))

    boot_stats = np.array(boot_stats)
    p_val = float(np.mean(np.abs(boot_stats) >= np.abs(obs_stat)))
    return obs_stat, p_val


# ---------------------------------------------------------------------------
# Rolling backtest engine
# ---------------------------------------------------------------------------

def rolling_var_backtest(
    returns: pd.Series,
    var_model_fn: Callable[[np.ndarray], float],
    confidence: float = 0.95,
    train_window: int = 252,
    step: int = 1,
) -> pd.DataFrame:
    """
    Rolling backtest: fit model trên window train, predict VaR t+1.

    Parameters
    ----------
    returns       : chuỗi return có index
    var_model_fn  : hàm f(returns_array) → var_float (giá trị âm)
    confidence    : mức tin cậy
    train_window  : kích thước cửa sổ train (rolling)
    step          : bước nhảy (1 = daily refit)

    Returns
    -------
    pd.DataFrame với cột:
        ['return', 'var_forecast', 'violation', 'rolling_viol_rate']
    """
    r = returns.dropna()
    n = len(r)
    results = []

    for t in range(train_window, n, step):
        train = r.iloc[t - train_window:t].values
        actual_ret = float(r.iloc[t])
        try:
            var_f = var_model_fn(train)
        except Exception:
            var_f = np.nan

        violation = actual_ret < var_f if not np.isnan(var_f) else False
        results.append({
            "date": r.index[t],
            "return": actual_ret,
            "var_forecast": var_f,
            "violation": violation,
        })

    df = pd.DataFrame(results).set_index("date")
    df["rolling_viol_rate"] = (
        df["violation"].rolling(min(train_window, len(df))).mean()
    )
    return df


# ---------------------------------------------------------------------------
# Full backtest pipeline
# ---------------------------------------------------------------------------

def backtest_var_model(
    returns: np.ndarray,
    var_forecasts: np.ndarray,
    model_name: str,
    confidence: float = 0.95,
) -> BacktestResult:
    """
    Chạy đầy đủ backtest cho một chuỗi VaR forecast.

    Parameters
    ----------
    returns       : return thực tế (n,)
    var_forecasts : VaR forecast (n,), giá trị âm
    model_name    : tên model
    confidence    : mức tin cậy

    Returns
    -------
    BacktestResult
    """
    r = np.asarray(returns)
    var = np.asarray(var_forecasts)
    n = len(r)

    violations = compute_violations(r, var)
    n_viol = int(violations.sum())
    viol_rate = n_viol / n
    expected_rate = 1 - confidence

    lr_uc, pval_uc = kupiec_test(violations, confidence)
    lr_ind, pval_ind = christoffersen_test(violations)
    if np.isnan(pval_ind):
        pval_ind = 1.0
        lr_ind = 0.0

    tl = basel_traffic_light(n_viol, n_obs=n, confidence=confidence)

    return BacktestResult(
        model_name=model_name,
        confidence=confidence,
        n_obs=n,
        n_violations=n_viol,
        violation_rate=viol_rate,
        expected_rate=expected_rate,
        kupiec_stat=lr_uc,
        kupiec_pvalue=pval_uc,
        christoffersen_stat=lr_ind,
        christoffersen_pvalue=pval_ind,
        traffic_light=tl,
        pass_kupiec=pval_uc > 0.05,
        pass_christoffersen=pval_ind > 0.05,
        violations_index=np.where(violations)[0],
    )


def backtest_multiple_models(
    returns: np.ndarray,
    model_forecasts: Dict[str, np.ndarray],
    confidence: float = 0.95,
) -> BacktestReport:
    """
    Backtest nhiều mô hình VaR cùng lúc.

    Parameters
    ----------
    returns         : return thực tế (n,)
    model_forecasts : dict {model_name: var_array}
    confidence      : mức tin cậy

    Returns
    -------
    BacktestReport
    """
    results = []
    for name, var_f in model_forecasts.items():
        result = backtest_var_model(returns, var_f, name, confidence)
        results.append(result)

    return BacktestReport(
        results=results,
        confidence=confidence,
        n_obs=len(returns),
    )


# ---------------------------------------------------------------------------
# Binomial test for violations
# ---------------------------------------------------------------------------

def binomial_var_test(
    n_violations: int,
    n_obs: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Binomial test đơn giản cho số lần violations.

    H0: P(violation) = 1 - confidence

    Phù hợp khi sample nhỏ.

    Returns
    -------
    Tuple[float, float] – (binomial_stat, p_value two-sided)
    """
    p_expected = 1 - confidence
    result = stats.binomtest(n_violations, n_obs, p=p_expected, alternative="two-sided")
    return float(n_violations / n_obs), float(result.pvalue)