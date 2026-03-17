"""
risk/metrics.py
----------------
Các chỉ số rủi ro tổng hợp của danh mục:

  - Sharpe / Sortino / Calmar / Omega ratio
  - VaR / CVaR wrappers (đọc từ models)
  - Tail Risk Measures: ETL, EVaR, Tail Gini
  - Portfolio risk decomposition: marginal / component / percentage
  - Risk-adjusted return metrics
  - RiskMetrics dataclass
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple, Union
from dataclasses import dataclass, field

from ..models.var_models import hs_var, parametric_var, garch_var
from ..models.cvar_models import historical_cvar, parametric_cvar
from ..models.evt import evt_var, evt_cvar


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RiskMetrics:
    """
    Tổng hợp toàn bộ chỉ số rủi ro của một danh mục / tài sản.

    Attributes
    ----------
    annualised_return : lợi nhuận hàng năm
    annualised_vol    : độ biến động hàng năm
    sharpe_ratio      : Sharpe ratio
    sortino_ratio     : Sortino ratio
    calmar_ratio      : Calmar ratio
    omega_ratio       : Omega ratio
    max_drawdown      : Maximum Drawdown
    var_95, var_99    : VaR 95%, 99%
    cvar_95, cvar_99  : CVaR 95%, 99%
    evt_var_99        : EVT-VaR 99%
    skewness          : skewness của return
    excess_kurtosis   : excess kurtosis
    tail_ratio        : |95th pct| / |5th pct|
    """
    annualised_return: float = np.nan
    annualised_vol: float = np.nan
    sharpe_ratio: float = np.nan
    sortino_ratio: float = np.nan
    calmar_ratio: float = np.nan
    omega_ratio: float = np.nan
    max_drawdown: float = np.nan
    var_95: float = np.nan
    var_99: float = np.nan
    cvar_95: float = np.nan
    cvar_99: float = np.nan
    evt_var_99: float = np.nan
    skewness: float = np.nan
    excess_kurtosis: float = np.nan
    tail_ratio: float = np.nan
    freq: int = 252
    extra: dict = field(default_factory=dict)

    def as_series(self) -> pd.Series:
        d = {k: v for k, v in self.__dict__.items() if k not in ("freq", "extra")}
        return pd.Series(d)

    def as_dataframe(self) -> pd.DataFrame:
        return self.as_series().to_frame("value")


# ---------------------------------------------------------------------------
# Return & volatility helpers
# ---------------------------------------------------------------------------

def _clean(returns: np.ndarray) -> np.ndarray:
    return np.asarray(returns)[~np.isnan(returns)]


def annualised_return(returns: np.ndarray, freq: int = 252) -> float:
    """Geometric annualised return."""
    r = _clean(returns)
    cum = float((1 + r).prod())
    n_years = len(r) / freq
    return float(cum ** (1 / n_years) - 1) if n_years > 0 else np.nan


def annualised_vol(returns: np.ndarray, freq: int = 252) -> float:
    """Annualised volatility (std × √freq)."""
    r = _clean(returns)
    return float(r.std(ddof=1) * np.sqrt(freq))


def downside_deviation(
    returns: np.ndarray,
    mar: float = 0.0,
    freq: int = 252,
) -> float:
    """
    Downside deviation (semi-deviation below MAR).

    Parameters
    ----------
    mar : Minimum Acceptable Return (default 0 = daily)
    """
    r = _clean(returns)
    downside = r[r < mar] - mar
    if len(downside) == 0:
        return 0.0
    return float(np.sqrt((downside**2).mean()) * np.sqrt(freq))


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum Drawdown từ chuỗi return."""
    r = _clean(returns)
    cum = (1 + r).cumprod()
    hwm = np.maximum.accumulate(cum)
    dd = (cum - hwm) / hwm
    return float(dd.min())


def drawdown_series(returns: np.ndarray) -> np.ndarray:
    """Chuỗi drawdown theo thời gian."""
    r = _clean(returns)
    cum = (1 + r).cumprod()
    hwm = np.maximum.accumulate(cum)
    return (cum - hwm) / hwm


# ---------------------------------------------------------------------------
# Risk-adjusted return ratios
# ---------------------------------------------------------------------------

def sharpe_ratio(
    returns: np.ndarray,
    risk_free: float = 0.0,
    freq: int = 252,
) -> float:
    """
    Sharpe ratio = (ann_return - rf) / ann_vol.

    Parameters
    ----------
    risk_free : annualised risk-free rate
    """
    ann_ret = annualised_return(returns, freq)
    ann_v = annualised_vol(returns, freq)
    if ann_v == 0:
        return np.nan
    return float((ann_ret - risk_free) / ann_v)


def sortino_ratio(
    returns: np.ndarray,
    risk_free: float = 0.0,
    mar: float = 0.0,
    freq: int = 252,
) -> float:
    """
    Sortino ratio = (ann_return - rf) / downside_deviation.

    Penalises hanya negative deviations.

    Parameters
    ----------
    mar : Minimum Acceptable Return (annualised, default 0)
    """
    ann_ret = annualised_return(returns, freq)
    dd = downside_deviation(returns, mar=mar / freq, freq=freq)
    if dd == 0:
        return np.nan
    return float((ann_ret - risk_free) / dd)


def calmar_ratio(
    returns: np.ndarray,
    freq: int = 252,
) -> float:
    """
    Calmar ratio = annualised_return / |max_drawdown|.

    Đo lường return so với drawdown risk.
    """
    ann_ret = annualised_return(returns, freq)
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return np.nan
    return float(ann_ret / mdd)


def omega_ratio(
    returns: np.ndarray,
    threshold: float = 0.0,
) -> float:
    """
    Omega ratio = E[max(R-L, 0)] / E[max(L-R, 0)].

    Tỷ lệ gain/loss so với ngưỡng L.

    Parameters
    ----------
    threshold : ngưỡng daily return (default 0)
    """
    r = _clean(returns)
    gains = np.maximum(r - threshold, 0).sum()
    losses = np.maximum(threshold - r, 0).sum()
    if losses == 0:
        return np.inf
    return float(gains / losses)


def information_ratio(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    freq: int = 252,
) -> float:
    """
    Information Ratio = ann_active_return / tracking_error.

    Parameters
    ----------
    portfolio_returns : chuỗi return danh mục
    benchmark_returns : chuỗi return benchmark
    """
    active = _clean(portfolio_returns) - _clean(benchmark_returns)
    ann_active = float(active.mean() * freq)
    te = float(active.std(ddof=1) * np.sqrt(freq))
    return float(ann_active / te) if te > 0 else np.nan


def tail_ratio(returns: np.ndarray, q: float = 0.05) -> float:
    """
    Tail ratio = |percentile(1-q)| / |percentile(q)|.

    > 1 → upside lớn hơn downside.
    """
    r = _clean(returns)
    upper = abs(np.quantile(r, 1 - q))
    lower = abs(np.quantile(r, q))
    return float(upper / lower) if lower != 0 else np.nan


def gain_to_pain_ratio(returns: np.ndarray) -> float:
    """
    Gain-to-Pain ratio = sum(positive returns) / |sum(negative returns)|.
    """
    r = _clean(returns)
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    return float(gains / losses) if losses > 0 else np.inf


# ---------------------------------------------------------------------------
# Tail-specific risk measures
# ---------------------------------------------------------------------------

def expected_tail_loss(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Expected Tail Loss (ETL) = CVaR lịch sử.

    Alias tiện dụng.
    """
    r = _clean(returns)
    return historical_cvar(r, confidence).cvar


def entropic_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    z: Optional[float] = None,
) -> float:
    """
    Entropic VaR (EVaR) – coherent risk measure dựa trên moment generating function.

    EVaR_α(X) = inf_{z>0} {z⁻¹ · ln(E[e^{-zX}] / (1-α))}

    Tighter upper bound hơn CVaR cho phân phối fat-tail.

    Parameters
    ----------
    confidence : mức tin cậy α
    z          : Lagrange multiplier (None → tối ưu hoá)
    """
    r = _clean(returns)
    alpha = 1 - confidence

    def objective(zz):
        if zz <= 0:
            return 1e10
        mgf = np.mean(np.exp(-zz * r))
        if mgf <= 0:
            return 1e10
        return (1 / zz) * (np.log(mgf) - np.log(alpha))

    if z is None:
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(objective, bounds=(1e-6, 100), method="bounded")
        z_opt = float(res.x)
    else:
        z_opt = z

    return float(objective(z_opt))


def tail_gini(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Tail Gini coefficient – đo lường bất bình đẳng trong phân phối đuôi.

    Tập trung vào sự phân tán của các quan sát vượt VaR.
    """
    r = _clean(returns)
    var = hs_var(r, confidence).var
    tail = np.sort(r[r <= var])
    n = len(tail)
    if n < 2:
        return np.nan
    indices = np.arange(1, n + 1)
    return float(2 * np.sum(indices * tail) / (n * tail.sum()) - (n + 1) / n)


# ---------------------------------------------------------------------------
# Portfolio risk decomposition
# ---------------------------------------------------------------------------

def marginal_var(
    weights: np.ndarray,
    returns_matrix: np.ndarray,
    confidence: float = 0.95,
    delta: float = 1e-4,
) -> np.ndarray:
    """
    Marginal VaR: ∂VaR_p / ∂w_i (numerical gradient).

    Parameters
    ----------
    weights        : trọng số danh mục (d,)
    returns_matrix : ma trận return (n × d)
    delta          : bước perturbation

    Returns
    -------
    np.ndarray (d,) – marginal VaR mỗi tài sản
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    R = np.asarray(returns_matrix)
    d = len(w)

    def port_var(ww):
        pr = R @ (ww / ww.sum())
        return hs_var(pr, confidence).var

    mvar = np.empty(d)
    base = port_var(w)
    for i in range(d):
        w_up = w.copy()
        w_up[i] += delta
        mvar[i] = (port_var(w_up) - base) / delta

    return mvar


def component_var(
    weights: np.ndarray,
    returns_matrix: np.ndarray,
    confidence: float = 0.95,
) -> pd.Series:
    """
    Component VaR = w_i × Marginal VaR_i.

    Tổng Component VaR ≈ Portfolio VaR (Euler allocation).

    Returns
    -------
    pd.Series (d,) – component VaR, tổng = portfolio VaR
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    mvar = marginal_var(w, returns_matrix, confidence)
    comp = w * mvar
    return pd.Series(comp, name="component_var")


def percentage_component_var(
    weights: np.ndarray,
    returns_matrix: np.ndarray,
    confidence: float = 0.95,
) -> pd.Series:
    """
    % đóng góp của mỗi tài sản vào VaR tổng.

    Returns
    -------
    pd.Series – % (tổng = 100%)
    """
    comp = component_var(weights, returns_matrix, confidence)
    total = comp.sum()
    if total == 0:
        return pd.Series(np.zeros(len(weights)))
    return (comp / total * 100).rename("pct_var_contribution")


def risk_parity_weights(
    returns_matrix: np.ndarray,
    confidence: float = 0.95,
    n_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Tính trọng số risk parity (Equal Risk Contribution).

    Mỗi tài sản đóng góp bằng nhau vào VaR danh mục.

    Parameters
    ----------
    n_iter : số vòng lặp tối đa
    tol    : điều kiện hội tụ

    Returns
    -------
    np.ndarray – trọng số (tổng = 1)
    """
    R = np.asarray(returns_matrix)
    d = R.shape[1]
    w = np.ones(d) / d

    for _ in range(n_iter):
        comp = component_var(w, R, confidence).values
        total = comp.sum()
        if total == 0:
            break
        target = total / d
        w_new = w * target / np.maximum(comp, 1e-10)
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    return w


# ---------------------------------------------------------------------------
# Full risk metrics summary
# ---------------------------------------------------------------------------

def compute_risk_metrics(
    returns: np.ndarray,
    risk_free: float = 0.0,
    freq: int = 252,
    compute_evt: bool = True,
) -> RiskMetrics:
    """
    Tính toàn bộ RiskMetrics từ một chuỗi return.

    Parameters
    ----------
    returns     : chuỗi return
    risk_free   : lãi suất phi rủi ro annualised
    freq        : số kỳ / năm
    compute_evt : True → bao gồm EVT-VaR (chậm hơn)

    Returns
    -------
    RiskMetrics dataclass
    """
    r = _clean(returns)

    ann_ret = annualised_return(r, freq)
    ann_v = annualised_vol(r, freq)
    mdd = max_drawdown(r)

    # VaR / CVaR
    var_95 = hs_var(r, 0.95).var
    var_99 = hs_var(r, 0.99).var
    cvar_95 = historical_cvar(r, 0.95).cvar
    cvar_99 = historical_cvar(r, 0.99).cvar

    # EVT-VaR
    evt_v99 = np.nan
    if compute_evt and len(r) >= 100:
        try:
            evt_v99 = evt_var(r, confidence=0.99)
        except Exception:
            pass

    return RiskMetrics(
        annualised_return=ann_ret,
        annualised_vol=ann_v,
        sharpe_ratio=sharpe_ratio(r, risk_free, freq),
        sortino_ratio=sortino_ratio(r, risk_free, freq=freq),
        calmar_ratio=calmar_ratio(r, freq),
        omega_ratio=omega_ratio(r),
        max_drawdown=mdd,
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        evt_var_99=evt_v99,
        skewness=float(stats.skew(r)),
        excess_kurtosis=float(stats.kurtosis(r, fisher=True)),
        tail_ratio=tail_ratio(r),
        freq=freq,
    )


def compare_assets_risk(
    returns_dict: dict,
    risk_free: float = 0.0,
    freq: int = 252,
) -> pd.DataFrame:
    """
    So sánh RiskMetrics của nhiều tài sản / chiến lược.

    Parameters
    ----------
    returns_dict : dict {name: returns_array}

    Returns
    -------
    pd.DataFrame – mỗi cột là một tài sản
    """
    rows = {}
    for name, ret in returns_dict.items():
        try:
            rm = compute_risk_metrics(ret, risk_free=risk_free, freq=freq)
            rows[name] = rm.as_series()
        except Exception as e:
            rows[name] = pd.Series({"error": str(e)})
    return pd.DataFrame(rows)