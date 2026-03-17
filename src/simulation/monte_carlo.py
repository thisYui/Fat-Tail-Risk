"""
simulation/monte_carlo.py
--------------------------
Engine Monte Carlo (MC) để mô phỏng phân phối P&L / return của danh mục:

  - MonteCarloEngine  : lớp chính chạy MC simulation
  - run_mc_prices     : sinh nhiều đường giá từ một ScenarioConfig
  - run_mc_returns    : sinh ma trận return MC
  - portfolio_mc      : MC trên danh mục nhiều tài sản với tương quan
  - summarise_mc      : tóm tắt kết quả MC (VaR, CVaR, phân vị, ...)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .generators import sample, multivariate_student_t, multivariate_normal
from .processes import gbm, merton_jump_diffusion, garch_11
from .scenarios import ScenarioConfig, CRISIS_SCENARIOS


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MCResult:
    """
    Lưu kết quả một lần chạy Monte Carlo.

    Attributes
    ----------
    paths       : np.ndarray (n_steps+1, n_paths) – price/value paths
    final_values: np.ndarray (n_paths,) – giá trị cuối kỳ
    pnl         : np.ndarray (n_paths,) – P&L = final - initial
    returns     : np.ndarray (n_paths,) – total return = pnl / initial
    scenario    : tên kịch bản
    n_paths     : số paths
    n_steps     : số bước
    """
    paths: np.ndarray
    final_values: np.ndarray
    pnl: np.ndarray
    returns: np.ndarray
    scenario: str
    n_paths: int
    n_steps: int

    def var(self, confidence: float = 0.95) -> float:
        """VaR tại mức tin cậy confidence."""
        return float(np.quantile(self.returns, 1 - confidence))

    def cvar(self, confidence: float = 0.95) -> float:
        """CVaR (Expected Shortfall) tại mức tin cậy confidence."""
        var = self.var(confidence)
        return float(self.returns[self.returns <= var].mean())

    def summary(self) -> pd.Series:
        """Tóm tắt toàn bộ kết quả MC."""
        return summarise_mc(self.returns)


# ---------------------------------------------------------------------------
# Core simulation functions
# ---------------------------------------------------------------------------

def run_mc_prices(
    scenario: Union[str, ScenarioConfig],
    n_paths: int = 10_000,
    n_steps: int = 252,
    S0: float = 100.0,
    dt: float = 1 / 252,
    seed: Optional[int] = None,
) -> MCResult:
    """
    Chạy Monte Carlo sinh các đường giá theo một kịch bản.

    Parameters
    ----------
    scenario : tên kịch bản (str) hoặc ScenarioConfig object
    n_paths  : số đường mô phỏng
    n_steps  : số bước thời gian (252 = 1 năm trading)
    S0       : giá ban đầu
    dt       : bước thời gian

    Returns
    -------
    MCResult
    """
    if isinstance(scenario, str):
        scenario = CRISIS_SCENARIOS[scenario]

    # Chọn process dựa trên cấu hình
    if scenario.jump_intensity > 0:
        paths = merton_jump_diffusion(
            n_steps=n_steps,
            n_paths=n_paths,
            mu=scenario.mu,
            sigma=scenario.sigma,
            lam=scenario.jump_intensity,
            jump_mu=scenario.jump_mu,
            jump_sigma=scenario.jump_sigma,
            S0=S0,
            dt=dt,
            seed=seed,
        )
    else:
        # GBM với innovation có thể từ fat-tail dist
        if scenario.dist == "normal":
            paths = gbm(
                n_steps=n_steps,
                n_paths=n_paths,
                mu=scenario.mu,
                sigma=scenario.sigma,
                S0=S0,
                dt=dt,
                seed=seed,
            )
        else:
            # Sinh innovations từ fat-tail distribution
            df = scenario.dist_params.get("df", 5)
            skew = scenario.dist_params.get("skew", 0.0)
            rng = np.random.default_rng(seed)

            if scenario.dist == "student_t":
                z = sample("student_t", n=n_steps * n_paths, df=df, seed=seed)
                # Standardise để giữ đúng sigma
                z = (z - z.mean()) / z.std()
            elif scenario.dist == "skewed_t":
                z = sample("skewed_t", n=n_steps * n_paths, df=df, skew=skew, seed=seed)
                z = (z - z.mean()) / z.std()
            else:
                z = rng.standard_normal(n_steps * n_paths)

            z = z.reshape(n_steps, n_paths)
            log_ret = (scenario.mu - 0.5 * scenario.sigma**2) * dt \
                      + scenario.sigma * np.sqrt(dt) * z
            log_price = np.vstack([
                np.zeros((1, n_paths)),
                np.cumsum(log_ret, axis=0),
            ])
            paths = S0 * np.exp(log_price)

    final = paths[-1]
    pnl = final - S0
    rets = (final - S0) / S0

    return MCResult(
        paths=paths,
        final_values=final,
        pnl=pnl,
        returns=rets,
        scenario=scenario.name,
        n_paths=n_paths,
        n_steps=n_steps,
    )


def run_mc_returns(
    scenario: Union[str, ScenarioConfig],
    n_paths: int = 10_000,
    n_steps: int = 252,
    dt: float = 1 / 252,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sinh ma trận log-return MC (n_steps × n_paths).

    Tiện dụng hơn run_mc_prices khi không cần chuỗi giá.

    Returns
    -------
    np.ndarray shape (n_steps, n_paths)
    """
    result = run_mc_prices(
        scenario=scenario,
        n_paths=n_paths,
        n_steps=n_steps,
        S0=1.0,
        dt=dt,
        seed=seed,
    )
    return np.log(result.paths[1:] / result.paths[:-1])


# ---------------------------------------------------------------------------
# Multi-asset portfolio MC
# ---------------------------------------------------------------------------

def portfolio_mc(
    weights: np.ndarray,
    mu_vec: np.ndarray,
    cov_matrix: np.ndarray,
    n_paths: int = 10_000,
    n_steps: int = 252,
    dist: str = "normal",
    df: float = 5.0,
    S0: float = 1.0,
    dt: float = 1 / 252,
    seed: Optional[int] = None,
) -> MCResult:
    """
    Monte Carlo trên danh mục nhiều tài sản với ma trận tương quan.

    Parameters
    ----------
    weights    : trọng số danh mục (d,), tổng = 1
    mu_vec     : vector drift (d,) – annualised
    cov_matrix : ma trận hiệp phương sai (d, d) – annualised
    dist       : 'normal' hoặc 'student_t'
    df         : bậc tự do nếu dist='student_t'

    Returns
    -------
    MCResult – P&L / return của danh mục
    """
    d = len(weights)
    weights = np.array(weights) / np.sum(weights)

    portfolio_values = np.full(n_paths, S0)
    all_paths = [portfolio_values.copy()]

    rng_seed = seed

    for t in range(n_steps):
        if dist == "normal":
            z = multivariate_normal(
                n=n_paths, mu=np.zeros(d), cov=cov_matrix, seed=rng_seed
            )
        else:
            z = multivariate_student_t(
                n=n_paths, df=df, mu=np.zeros(d), cov=cov_matrix, seed=rng_seed
            )
        if rng_seed is not None:
            rng_seed += 1

        # Log return của từng tài sản tại bước t
        sigmas = np.sqrt(np.diag(cov_matrix))
        log_ret = (mu_vec - 0.5 * sigmas**2) * dt + np.sqrt(dt) * z  # (n_paths, d)
        asset_returns = np.exp(log_ret) - 1                           # simple return
        port_return = asset_returns @ weights                          # (n_paths,)
        portfolio_values = portfolio_values * (1 + port_return)
        all_paths.append(portfolio_values.copy())

    paths_array = np.array(all_paths)  # (n_steps+1, n_paths)
    final = paths_array[-1]
    pnl = final - S0
    rets = (final - S0) / S0

    return MCResult(
        paths=paths_array,
        final_values=final,
        pnl=pnl,
        returns=rets,
        scenario=f"portfolio_{dist}",
        n_paths=n_paths,
        n_steps=n_steps,
    )


# ---------------------------------------------------------------------------
# Multi-scenario comparison
# ---------------------------------------------------------------------------

def run_scenario_comparison(
    scenario_names: List[str],
    n_paths: int = 10_000,
    n_steps: int = 252,
    S0: float = 100.0,
    dt: float = 1 / 252,
    seed: Optional[int] = None,
) -> Dict[str, MCResult]:
    """
    Chạy MC cho nhiều kịch bản cùng lúc và trả về dict kết quả.

    Parameters
    ----------
    scenario_names : danh sách tên kịch bản (xem CRISIS_SCENARIOS.keys())

    Returns
    -------
    dict {scenario_name: MCResult}
    """
    results = {}
    for i, name in enumerate(scenario_names):
        sc_seed = (seed + i * 1000) if seed is not None else None
        results[name] = run_mc_prices(
            scenario=name,
            n_paths=n_paths,
            n_steps=n_steps,
            S0=S0,
            dt=dt,
            seed=sc_seed,
        )
    return results


def compare_scenarios_summary(
    results: Dict[str, MCResult],
    confidence_levels: Tuple[float, ...] = (0.90, 0.95, 0.99),
) -> pd.DataFrame:
    """
    So sánh các kịch bản theo VaR, CVaR, mean, std, min.

    Parameters
    ----------
    results : output của run_scenario_comparison
    confidence_levels : các mức tin cậy

    Returns
    -------
    pd.DataFrame với index = scenario name
    """
    rows = []
    for name, res in results.items():
        row = {"scenario": name}
        row["mean_return"] = float(res.returns.mean())
        row["std_return"] = float(res.returns.std())
        row["min_return"] = float(res.returns.min())
        row["max_return"] = float(res.returns.max())
        for cl in confidence_levels:
            pct = int(cl * 100)
            row[f"var_{pct}"] = res.var(cl)
            row[f"cvar_{pct}"] = res.cvar(cl)
        rows.append(row)
    return pd.DataFrame(rows).set_index("scenario")


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def summarise_mc(
    returns: np.ndarray,
    confidence_levels: Tuple[float, ...] = (0.90, 0.95, 0.99),
) -> pd.Series:
    """
    Tóm tắt phân phối return MC.

    Parameters
    ----------
    returns : array 1D – total return của mỗi path

    Returns
    -------
    pd.Series: mean, std, skew, kurt, min, max,
               var_{90,95,99}, cvar_{90,95,99},
               pct_{1,5,10,25,50,75,90,95,99}
    """
    from scipy import stats as sp_stats

    r = np.asarray(returns)
    result = {
        "mean": float(r.mean()),
        "std": float(r.std()),
        "skewness": float(sp_stats.skew(r)),
        "excess_kurtosis": float(sp_stats.kurtosis(r)),
        "min": float(r.min()),
        "max": float(r.max()),
    }

    for cl in confidence_levels:
        pct = int(cl * 100)
        var_val = float(np.quantile(r, 1 - cl))
        cvar_val = float(r[r <= var_val].mean()) if (r <= var_val).any() else var_val
        result[f"var_{pct}"] = var_val
        result[f"cvar_{pct}"] = cvar_val

    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        result[f"pct_{p}"] = float(np.percentile(r, p))

    return pd.Series(result)


# ---------------------------------------------------------------------------
# MonteCarloEngine – class wrapper
# ---------------------------------------------------------------------------

class MonteCarloEngine:
    """
    Wrapper tiện lợi để chạy và cache các kết quả Monte Carlo.

    Example
    -------
    >>> engine = MonteCarloEngine(n_paths=50_000, n_steps=252, seed=42)
    >>> result = engine.run("gfc_2008")
    >>> print(result.var(0.99))
    >>> summary = engine.compare(["base", "gfc_2008", "covid_2020"])
    """

    def __init__(
        self,
        n_paths: int = 10_000,
        n_steps: int = 252,
        S0: float = 100.0,
        dt: float = 1 / 252,
        seed: Optional[int] = None,
    ):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.S0 = S0
        self.dt = dt
        self.seed = seed
        self._cache: Dict[str, MCResult] = {}

    def run(
        self,
        scenario: Union[str, ScenarioConfig],
        use_cache: bool = True,
    ) -> MCResult:
        """
        Chạy MC cho một kịch bản (cache kết quả theo tên).

        Parameters
        ----------
        scenario  : tên kịch bản hoặc ScenarioConfig
        use_cache : True → trả về kết quả đã tính nếu có
        """
        name = scenario if isinstance(scenario, str) else scenario.name
        if use_cache and name in self._cache:
            return self._cache[name]

        result = run_mc_prices(
            scenario=scenario,
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            S0=self.S0,
            dt=self.dt,
            seed=self.seed,
        )
        self._cache[name] = result
        return result

    def compare(
        self,
        scenario_names: List[str],
        confidence_levels: Tuple[float, ...] = (0.90, 0.95, 0.99),
    ) -> pd.DataFrame:
        """
        So sánh nhiều kịch bản, trả về bảng tổng hợp.

        Parameters
        ----------
        scenario_names    : danh sách tên kịch bản
        confidence_levels : các mức tin cậy VaR/CVaR
        """
        results = {name: self.run(name) for name in scenario_names}
        return compare_scenarios_summary(results, confidence_levels)

    def clear_cache(self) -> None:
        """Xóa toàn bộ cache."""
        self._cache.clear()

    @property
    def cached_scenarios(self) -> List[str]:
        """Danh sách các kịch bản đã được cache."""
        return list(self._cache.keys())