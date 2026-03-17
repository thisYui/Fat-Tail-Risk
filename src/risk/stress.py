"""
risk/stress.py
---------------
Stress testing framework cho danh mục đầu tư:

  - Historical stress scenarios (replay lịch sử)
  - Hypothetical / parametric scenarios
  - Factor-based stress testing
  - Reverse stress testing (tìm ngưỡng phá sản)
  - Sensitivity analysis
  - StressResult & StressReport dataclasses
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field

from ..simulation.scenarios import (
    ScenarioConfig,
    FactorShock,
    CRISIS_SCENARIOS,
    FACTOR_SHOCKS,
    bootstrap_scenarios,
    worst_scenarios,
)
from ..models.var_models import hs_var
from ..models.cvar_models import historical_cvar


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class StressResult:
    """
    Kết quả một bài kiểm thử stress.

    Attributes
    ----------
    scenario_name   : tên kịch bản
    pnl             : P&L ước tính (giá trị tiền hoặc %)
    return_estimate : return ước tính
    var_breach      : True nếu loss vượt VaR hiện tại
    notes           : ghi chú
    """
    scenario_name: str
    pnl: float
    return_estimate: float
    var_breach: bool = False
    cvar_breach: bool = False
    notes: str = ""
    details: dict = field(default_factory=dict)

    def as_series(self) -> pd.Series:
        return pd.Series({
            "scenario": self.scenario_name,
            "pnl": self.pnl,
            "return_estimate": self.return_estimate,
            "var_breach": self.var_breach,
            "cvar_breach": self.cvar_breach,
            "notes": self.notes,
        })


@dataclass
class StressReport:
    """
    Báo cáo tổng hợp toàn bộ bài stress test.
    """
    results: List[StressResult]
    portfolio_value: float
    current_var_95: float
    current_var_99: float
    current_cvar_95: float

    def summary(self) -> pd.DataFrame:
        rows = [r.as_series() for r in self.results]
        df = pd.DataFrame(rows).set_index("scenario")
        df["pnl_pct"] = df["pnl"] / self.portfolio_value * 100
        return df.sort_values("pnl")

    def worst_scenarios(self, n: int = 5) -> pd.DataFrame:
        return self.summary().head(n)

    def scenarios_breaching_var(self) -> pd.DataFrame:
        df = self.summary()
        return df[df["var_breach"]]


# ---------------------------------------------------------------------------
# Historical stress replay
# ---------------------------------------------------------------------------

def historical_stress_test(
    portfolio_returns: pd.Series,
    portfolio_weights: Optional[np.ndarray] = None,
    asset_returns: Optional[pd.DataFrame] = None,
    periods: Optional[Dict[str, Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """
    Replay các giai đoạn khủng hoảng lịch sử lên danh mục.

    Parameters
    ----------
    portfolio_returns : chuỗi return danh mục (DatetimeIndex)
    portfolio_weights : trọng số (nếu muốn áp lên asset_returns)
    asset_returns     : ma trận return các tài sản (DatetimeIndex)
    periods           : dict {name: (start_date, end_date)} (None = dùng mặc định)

    Returns
    -------
    pd.DataFrame – kết quả mỗi giai đoạn
    """
    DEFAULT_PERIODS = {
        "Dot-com crash (2000-2002)": ("2000-03-01", "2002-10-31"),
        "GFC onset (2008-H2)": ("2008-07-01", "2009-03-31"),
        "European debt crisis (2011)": ("2011-07-01", "2011-12-31"),
        "China crash (2015-2016)": ("2015-06-01", "2016-02-29"),
        "COVID crash (2020-Q1)": ("2020-02-01", "2020-03-31"),
        "Ukraine war shock (2022-Q1)": ("2022-01-01", "2022-03-31"),
        "Rate hike selloff (2022)": ("2022-01-01", "2022-12-31"),
    }

    if periods is None:
        periods = DEFAULT_PERIODS

    r = portfolio_returns.dropna()
    rows = []

    for name, (start, end) in periods.items():
        try:
            window = r.loc[start:end]
            if len(window) < 3:
                continue
            total_ret = float((1 + window).prod() - 1)
            max_dd = float(((1 + window).cumprod() / (1 + window).cumprod().cummax() - 1).min())
            vol = float(window.std() * np.sqrt(252))
            rows.append({
                "scenario": name,
                "start": start,
                "end": end,
                "total_return": total_ret,
                "max_drawdown": max_dd,
                "annualised_vol": vol,
                "n_days": len(window),
            })
        except Exception:
            continue

    return pd.DataFrame(rows).set_index("scenario")


# ---------------------------------------------------------------------------
# Parametric scenario stress test
# ---------------------------------------------------------------------------

def parametric_stress_test(
    returns: np.ndarray,
    portfolio_value: float = 1_000_000.0,
    confidence: float = 0.95,
    scenarios: Optional[Dict[str, ScenarioConfig]] = None,
) -> StressReport:
    """
    Stress test bằng các kịch bản parametric.

    Với mỗi kịch bản, ước tính P&L bằng:
    E[return | scenario] = μ_scenario × horizon

    Parameters
    ----------
    returns         : chuỗi return lịch sử (để tính VaR hiện tại)
    portfolio_value : giá trị danh mục (đơn vị tiền)
    confidence      : mức tin cậy cho VaR breach check
    scenarios       : dict kịch bản (None = dùng CRISIS_SCENARIOS)

    Returns
    -------
    StressReport
    """
    if scenarios is None:
        scenarios = CRISIS_SCENARIOS

    r = np.asarray(returns)[~np.isnan(returns)]
    var_95 = hs_var(r, 0.95).var
    var_99 = hs_var(r, 0.99).var
    cvar_95 = historical_cvar(r, 0.95).cvar

    results = []
    for name, sc in scenarios.items():
        # Ước tính return kịch bản: annual mu rescale sang monthly (~21 ngày)
        ret_est = sc.mu / 12   # 1-month scenario
        # Điều chỉnh cho jump component
        if sc.jump_intensity > 0:
            expected_jump = sc.jump_intensity / 12 * sc.jump_mu
            ret_est += expected_jump

        pnl = ret_est * portfolio_value
        results.append(StressResult(
            scenario_name=name,
            pnl=pnl,
            return_estimate=ret_est,
            var_breach=ret_est < var_95,
            cvar_breach=ret_est < cvar_95,
            notes=sc.description,
            details={"mu": sc.mu, "sigma": sc.sigma, "jump_lam": sc.jump_intensity},
        ))

    return StressReport(
        results=results,
        portfolio_value=portfolio_value,
        current_var_95=var_95,
        current_var_99=var_99,
        current_cvar_95=cvar_95,
    )


# ---------------------------------------------------------------------------
# Factor-based stress test
# ---------------------------------------------------------------------------

def factor_stress_test(
    returns: np.ndarray,
    factor_returns: pd.DataFrame,
    portfolio_value: float = 1_000_000.0,
    shocks: Optional[Dict[str, FactorShock]] = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Stress test dựa trên factor model.

    P&L ≈ Σ β_i × shock_i × portfolio_value

    Bước 1: Ước lượng factor betas bằng OLS
    Bước 2: Áp dụng factor shocks
    Bước 3: Tính P&L ước tính

    Parameters
    ----------
    returns         : chuỗi return danh mục (n,)
    factor_returns  : ma trận factor returns (n × k)
    portfolio_value : giá trị danh mục
    shocks          : dict {name: FactorShock}

    Returns
    -------
    pd.DataFrame – mỗi hàng là một shock scenario
    """
    from sklearn.linear_model import LinearRegression

    r = np.asarray(returns)[~np.isnan(returns)]
    F = factor_returns.iloc[:len(r)].values

    # OLS factor betas
    reg = LinearRegression(fit_intercept=True)
    reg.fit(F, r)
    betas = reg.coef_
    factor_names = factor_returns.columns.tolist()
    beta_dict = dict(zip(factor_names, betas))

    if shocks is None:
        shocks = FACTOR_SHOCKS

    rows = []
    for shock_name, fs in shocks.items():
        pnl_pct = sum(
            beta_dict.get(f, 0) * sv
            for f, sv in fs.shocks.items()
        )
        pnl_abs = pnl_pct * portfolio_value
        rows.append({
            "shock": shock_name,
            "pnl_pct": pnl_pct,
            "pnl_abs": pnl_abs,
            "description": fs.description,
        })

    return pd.DataFrame(rows).set_index("shock").sort_values("pnl_abs")


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    returns: np.ndarray,
    param_name: str,
    param_values: np.ndarray,
    risk_fn: Callable[[np.ndarray, float], float],
) -> pd.DataFrame:
    """
    Phân tích nhạy cảm của một chỉ số rủi ro theo tham số.

    Parameters
    ----------
    returns      : chuỗi return
    param_name   : tên tham số (vd 'confidence', 'window')
    param_values : mảng giá trị tham số cần thử
    risk_fn      : hàm f(returns, param) → float

    Returns
    -------
    pd.DataFrame với cột [param_name, 'risk_value']
    """
    r = np.asarray(returns)[~np.isnan(returns)]
    rows = []
    for v in param_values:
        try:
            risk_val = risk_fn(r, v)
            rows.append({param_name: v, "risk_value": risk_val})
        except Exception:
            rows.append({param_name: v, "risk_value": np.nan})
    return pd.DataFrame(rows)


def var_confidence_sensitivity(
    returns: np.ndarray,
    confidence_levels: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    VaR / CVaR theo nhiều mức tin cậy.

    Returns
    -------
    pd.DataFrame với cột ['confidence', 'var', 'cvar', 'var_cvar_ratio']
    """
    r = np.asarray(returns)[~np.isnan(returns)]
    if confidence_levels is None:
        confidence_levels = np.arange(0.80, 0.9999, 0.01)

    rows = []
    for cl in confidence_levels:
        var = hs_var(r, cl).var
        cvar = historical_cvar(r, cl).cvar
        rows.append({
            "confidence": cl,
            "var": var,
            "cvar": cvar,
            "var_cvar_ratio": cvar / var if var != 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reverse stress testing
# ---------------------------------------------------------------------------

def reverse_stress_test(
    returns: np.ndarray,
    portfolio_value: float,
    loss_threshold: float,
    method: str = "historical",
) -> dict:
    """
    Reverse stress test: tìm kịch bản / ngưỡng dẫn đến lỗ vượt threshold.

    Thay vì hỏi "nếu X xảy ra thì thiệt hại bao nhiêu?",
    hỏi "cần return bao nhiêu để thiệt hại = loss_threshold?"

    Parameters
    ----------
    portfolio_value : giá trị danh mục hiện tại
    loss_threshold  : mức lỗ tuyệt đối cần tìm (số dương)
    method          : 'historical' (dùng empirical CDF)

    Returns
    -------
    dict với:
        required_return : mức return cần thiết (âm)
        implied_var_level : mức tin cậy tương đương
        historical_frequency : tần suất lịch sử vượt ngưỡng
        worst_10_scenarios : top 10 ngày tệ nhất từ lịch sử
    """
    r = np.asarray(returns)[~np.isnan(returns)]
    required_ret = -loss_threshold / portfolio_value

    # Implied confidence level: P(R ≤ required_ret)
    implied_alpha = float(np.mean(r <= required_ret))
    implied_confidence = 1 - implied_alpha

    # Historical frequency
    hist_freq = implied_alpha

    # Worst historical scenarios
    worst_idx = np.argsort(r)[:10]
    worst_returns = r[worst_idx]

    return {
        "required_return": required_ret,
        "loss_threshold": loss_threshold,
        "portfolio_value": portfolio_value,
        "implied_var_confidence": implied_confidence,
        "historical_frequency": hist_freq,
        "expected_days_between_events": 1 / hist_freq if hist_freq > 0 else np.inf,
        "worst_10_returns": worst_returns.tolist(),
    }


# ---------------------------------------------------------------------------
# Monte Carlo stress test
# ---------------------------------------------------------------------------

def monte_carlo_stress_summary(
    returns: np.ndarray,
    n_simulations: int = 10_000,
    horizon: int = 21,
    confidence_levels: Tuple[float, ...] = (0.95, 0.99),
    seed: Optional[int] = None,
) -> pd.Series:
    """
    Stress test bằng cách bootstrap và simulate forward.

    Bước 1: Bootstrap window=horizon từ lịch sử n_simulations lần
    Bước 2: Tính total return mỗi path
    Bước 3: Tóm tắt phân phối

    Parameters
    ----------
    horizon : số ngày mỗi simulation path

    Returns
    -------
    pd.Series – mean, std, VaR/CVaR nhiều mức, worst/best case
    """
    df_bootstrap = bootstrap_scenarios(
        pd.Series(returns),
        n_scenarios=n_simulations,
        window=horizon,
        seed=seed,
    )
    path_returns = (1 + df_bootstrap).prod(axis=1) - 1
    r = path_returns.values

    result = {
        "horizon_days": horizon,
        "n_simulations": n_simulations,
        "mean_return": float(r.mean()),
        "std_return": float(r.std()),
        "min_return": float(r.min()),
        "max_return": float(r.max()),
        "pct_negative": float(np.mean(r < 0)),
    }
    for cl in confidence_levels:
        pct = int(cl * 100)
        var = float(np.quantile(r, 1 - cl))
        cvar = float(r[r <= var].mean()) if (r <= var).any() else var
        result[f"var_{pct}"] = var
        result[f"cvar_{pct}"] = cvar

    return pd.Series(result)